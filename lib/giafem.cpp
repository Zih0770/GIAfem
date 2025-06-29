#include "giafem.hpp"
#include <cmath>
#include <iostream>

using namespace std;
using namespace mfem;

namespace giafem
{
//Utilities

//Material Models
ElasticityModel ParseElasticityModel(const char *str)
{
    if (strcmp(str, "linear") == 0) return ElasticityModel::linear;
    if (strcmp(str, "neo-hookean") == 0) return ElasticityModel::neoHookean;

    //mfem::err << "Unknown elasticity model: " << str << std::endl;
    MFEM_ABORT("Invalid elasticity model.");
    return ElasticityModel::linear;  // unreachable
}

RheologyModel ParseRheologyModel(const char *str)
{
    if (strcmp(str, "Maxwell") == 0) return RheologyModel::Maxwell;
    if (strcmp(str, "Maxwell_nonlinear") == 0) return RheologyModel::Maxwell_nonlinear;
    if (strcmp(str, "Kelvin-Voigt") == 0) return RheologyModel::KelvinVoigt;

    MFEM_ABORT("Invalid rheology model.");
    return RheologyModel::Maxwell;
}


//Operators
ViscoelasticOperator::ViscoelasticOperator(ParFiniteElementSpace &fes_u_, ParFiniteElementSpace &fes_m_, ParFiniteElementSpace &fes_properties_, ParFiniteElementSpace &fes_w_, 
                                           ParGridFunction &u_gf_, ParGridFunction &m_gf_, ParGridFunction &d_gf_, 
                                           Coefficient &lamb_, Coefficient &mu_, Coefficient &tau_, Coefficient &loading_,
                                           const real_t rel_tol_, const real_t implicit_scheme_res_,
                                           const char *elasticity_model_str, const char *rheology_model_str)   
                                           : TimeDependentOperator(fes_m_.GetTrueVSize(), (real_t) 0.0), 
                                           fes_u(fes_u_), fes_m(fes_m_), fes_properties(fes_properties_), fes_w(fes_w_), 
                                           u_gf(u_gf_), m_gf(m_gf_), d_gf(d_gf_), lamb_gf(&fes_properties), mu_gf(&fes_properties), tau_gf(&fes_properties), 
                                           lamb(lamb_), mu(mu_), tau(tau_), loading(loading_), 
                                           B(&fes_m, &fes_u), current_dt(0.0), Dev(&fes_u_, &fes_m_), 
                                           K_solver(fes_u_.GetComm()), rigid_solver(fes_u_.GetComm(), &fes_u_),
                                           rel_tol(rel_tol_), implicit_scheme_res(implicit_scheme_res_)
{
    comm = fes_u.GetComm();
    dim = fes_u.GetMesh()->SpaceDimension();
    EM = ParseElasticityModel(elasticity_model_str);
    RM = ParseRheologyModel(rheology_model_str);
    tau_gf.ProjectCoefficient(tau);
    tau_gf.GetTrueDofs(tau_vec);
    Array<int> ess_bdr(fes_u.GetMesh()->bdr_attributes.Max());
    ess_bdr = 0;
    fes_u.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    K = new ParBilinearForm(&fes_u);
    switch (EM)
    {
        case ElasticityModel::linear:
            {
                K->AddDomainIntegrator(new ElasticityIntegrator(lamb, mu));
                break;
            }
        default:
            MFEM_ABORT("Unhandled elasticity model.");
    }
    K->SetOperatorType(Operator::PETSC_MATAIJ);
    K->Assemble();
    K->Finalize();

    //K->FormSystemMatrix(ess_tdof_list, *hK);
    hK = K->ParallelAssemble();
    Kmat = new PetscParMatrix(hK, Operator::PETSC_MATAIJ);

    K_prec = new PetscPreconditioner(*Kmat, "solver_");
    K_solver.iterative_mode = true;
    K_solver.SetTol(rel_tol);
    K_solver.SetMaxIter(500);
    K_solver.SetPrintLevel(0);
    K_solver.SetOperator(*Kmat);
    K_solver.SetPreconditioner(*K_prec);
    rigid_solver.SetSolver(K_solver);
    switch (RM)
    {
        case RheologyModel::Maxwell:
            {
                B.AddDomainIntegrator(new ViscoelasticForcing(mu));
                break;
            }
        default:
            MFEM_ABORT("Unhandled rheology model.");
    }
    B.Assemble();

    Dev.AddDomainInterpolator(new DevStrainInterpolator);
    Dev.Assemble();
    Dev.Finalize();
}

void ViscoelasticOperator::Mult(const Vector &m_vec, Vector &dm_dt_vec) const
{
    ParLinearForm *b = new ParLinearForm(&fes_u);
    //m_gf.SetFromTrueDofs(m_vec);
    //VectorGridFunctionCoefficient m_coeff(&m_gf);
    //b->AddDomainIntegrator(new ViscoelasticRHSIntegrator(mu, m_coeff));
    b->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(loading));
    b->Assemble();
    //b->ParallelAssemble(b_vec);
    m_gf.SetFromTrueDofs(m_vec);
    B.AddMult(m_gf, *b);
    K->FormLinearSystem(ess_tdof_list, u_gf, *b, *Kmat, x_vec, b_vec);
    Kmat->SetBlockSize(dim);

    rigid_solver.Mult(b_vec, x_vec);
    //Kmat->Mult(b_vec, x_vec);
    K->RecoverFEMSolution(x_vec, *b, u_gf);
    u_gf.SetTrueVector();

    Dev.Mult(u_gf, d_gf);
    d_gf.GetTrueDofs(d_vec);
    for (int i = 0; i < d_vec.Size(); i++)
    {
        dm_dt_vec[i] = (d_vec[i] - m_vec[i]) / tau_vec[i % tau_vec.Size()];
        //real_t tau_val = 1000 * 3.15576e7;
        //dm_dt_vec[i] = (d_vec[i] - m_vec[i]) / tau_val;
    }

    delete b;
}

void ViscoelasticOperator::ImplicitSolve(const real_t dt, const Vector &m_vec, Vector &dm_dt_vec)
{
    current_dt = dt;
    this->Mult(m_vec, dm_dt_vec);
    Vector dm_dt_old;
    Vector res;
    int iter = 0;
    do {
    dm_dt_old = dm_dt_vec;
    Vector m_est;
    //add(m_vec, current_dt, dm_dt_vec, m_est);
    m_est = dm_dt_vec; m_est *= current_dt; m_est += m_vec;
    this->Mult(m_est, dm_dt_vec);
    for (int i = 0; i < dm_dt_vec.Size(); i++)
    {
        dm_dt_vec[i] *= tau_vec[i % tau_vec.Size()];
        dm_dt_vec[i] /= tau_vec[i % tau_vec.Size()] + current_dt;
    }
    res = dm_dt_vec.Add(-1.0, dm_dt_old);
    } while (res.Norml2()/dm_dt_old.Norml2() > implicit_scheme_res);

}

void ViscoelasticOperator::CalcStrainEnergyDensity(ParGridFunction &w_gf) const
{
    StrainEnergyCoefficient w_coeff(u_gf, lamb, mu);
    w_gf.ProjectCoefficient(w_coeff);
}


ExtTrialMixedBilinearForm::ExtTrialMixedBilinearForm (FiniteElementSpace *tr_fes,
        FiniteElementSpace *te_fes,
        FiniteElementSpace *tr_fes_cond, SubMesh *mesh_cond)
    : MixedBilinearForm(tr_fes, te_fes)
{
    trial_fes_cond = tr_fes_cond;
    from = mesh_cond->GetFrom();
    parent_element_ids = mesh_cond->GetParentElementIDMap(); //&
}

void ExtTrialMixedBilinearForm::Assemble(int skip_zeros)
{
    /*if (ext)
    {
        ext->Assemble();
        return;
    }*/

    ElementTransformation *eltrans;
    DofTransformation *dom_dof_trans;
    DofTransformation *ran_dof_trans;
    DenseMatrix elmat;

    Mesh *mesh = trial_fes_cond -> GetMesh();

    if (mat == NULL)
    {
        mat = new SparseMatrix(height, width);
    }

    if (domain_integs.Size())
    {
        for (int k = 0; k < domain_integs.Size(); k++)
        {
            if (domain_integs_marker[k] != NULL)
            {
                MFEM_VERIFY(domain_integs_marker[k]->Size() ==
                        (mesh->attributes.Size() ? mesh->attributes.Max() : 0),
                        "invalid element marker for domain integrator #"
                        << k << ", counting from zero");
            }
        }

        for (int i = 0; i < trial_fes_cond -> GetNE(); i++)
        {
            //const int elem_attr = mesh->GetAttribute(i);
            dom_dof_trans = trial_fes_cond -> GetElementVDofs (i, trial_vdofs);
            ran_dof_trans = test_fes  -> GetElementVDofs (i, test_vdofs);
            eltrans = trial_fes_cond -> GetElementTransformation (i);

            elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
            elmat = 0.0;
            for (int k = 0; k < domain_integs.Size(); k++)
            {
                //if (domain_integs_marker[k] == NULL ||
                //    (*(domain_integs_marker[k]))[elem_attr-1] == 1)
                {
                    domain_integs[k] -> AssembleElementMatrix2 (*trial_fes_cond -> GetFE(i),
                                                                *test_fes  -> GetFE(i),
                                                                *eltrans, elemmat);
                    elmat += elemmat;
                }
            }
            if (ran_dof_trans || dom_dof_trans)
            {
                TransformDual(ran_dof_trans, dom_dof_trans, elmat);
            }
            if (!vdof_to_vdof_map)
            {
                vdof_to_vdof_map = new Array<int>();
                SubMeshUtils::BuildVdofToVdofMap(*trial_fes_cond, *trial_fes, from, parent_element_ids, *vdof_to_vdof_map);
            }

            Array<int> trial_vdofs_ext(trial_vdofs.Size());
            for (int l = 0; l < trial_vdofs.Size(); l++)
                trial_vdofs_ext[l] = (*vdof_to_vdof_map)[trial_vdofs[l]];

            mat->AddSubMatrix(test_vdofs, trial_vdofs_ext, elmat, skip_zeros);
        }
    }
}



ExtTestMixedBilinearForm::ExtTestMixedBilinearForm (FiniteElementSpace *tr_fes,
                                                    FiniteElementSpace *te_fes,
                                                    FiniteElementSpace *te_fes_cond, SubMesh *mesh_cond)
    : MixedBilinearForm(tr_fes, te_fes)
{
   test_fes_cond = te_fes_cond;
   from = mesh_cond->GetFrom();
   parent_element_ids = mesh_cond->GetParentElementIDMap();
}

void ExtTestMixedBilinearForm::Assemble(int skip_zeros)
{
    /*if (ext)
    {
        ext->Assemble();
        return;
    }*/

    ElementTransformation *eltrans;
    DofTransformation *dom_dof_trans;
    DofTransformation *ran_dof_trans;
    DenseMatrix elmat;

    Mesh *mesh = test_fes_cond -> GetMesh();

    if (mat == NULL)
    {
        mat = new SparseMatrix(height, width);
    }

    if (domain_integs.Size())
    {
        for (int k = 0; k < domain_integs.Size(); k++)
        {
            if (domain_integs_marker[k] != NULL)
            {
                MFEM_VERIFY(domain_integs_marker[k]->Size() ==
                        (mesh->attributes.Size() ? mesh->attributes.Max() : 0),
                        "invalid element marker for domain integrator #"
                        << k << ", counting from zero");
            }
        }

        for (int i = 0; i < test_fes_cond -> GetNE(); i++)
        {
            //const int elem_attr = mesh->GetAttribute(i);
            dom_dof_trans = trial_fes -> GetElementVDofs (i, trial_vdofs);
            ran_dof_trans = test_fes_cond  -> GetElementVDofs (i, test_vdofs);
            eltrans = test_fes_cond -> GetElementTransformation (i);

            elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
            elmat = 0.0;
            for (int k = 0; k < domain_integs.Size(); k++)
            {
                //if (domain_integs_marker[k] == NULL ||
                //    (*(domain_integs_marker[k]))[elem_attr-1] == 1)
                {
                    domain_integs[k] -> AssembleElementMatrix2 (*trial_fes -> GetFE(i),
                                                                *test_fes_cond  -> GetFE(i),
                                                                *eltrans, elemmat);
                    elmat += elemmat;
                }
            }
            if (ran_dof_trans || dom_dof_trans)
            {
                TransformDual(ran_dof_trans, dom_dof_trans, elmat); //
            }
            if (!vdof_to_vdof_map)
            {
                vdof_to_vdof_map = new Array<int>();
                SubMeshUtils::BuildVdofToVdofMap(*test_fes_cond, *test_fes, from, parent_element_ids, *vdof_to_vdof_map);
            }

            Array<int> test_vdofs_ext(test_vdofs.Size());
            for (int l = 0; l < test_vdofs.Size(); l++)
                test_vdofs_ext[l] = (*vdof_to_vdof_map)[test_vdofs[l]];

            mat->AddSubMatrix(test_vdofs_ext, trial_vdofs, elmat, skip_zeros);
        }
    }
}


//Interpolators
void GradInterpolator::AssembleElementMatrix2(const FiniteElement &u_fe,
                                                const FiniteElement &e_fe,
                                                ElementTransformation &Trans,
                                                DenseMatrix &elmat)
{
    //MFEM_ASSERT(u_fe.GetMapType() == VALUE, "");
    int N = u_fe.GetDof(), M = e_fe.GetDof();
 
    DenseMatrix dshape(N, dim);
 
    if (mode == VecMode::REDUCED) {
    elmat.SetSize(vdim*M, dim*N);
    } else {
    elmat.SetSize(dim*dim*M, dim*N); }
    elmat = 0.0;

    if (mode == VecMode::REDUCED)
    {
        for (int m = 0; m < M; m++)
        {
            const auto &ip  = e_fe.GetNodes().IntPoint(m);    
            Trans.SetIntPoint(&ip);
            u_fe.CalcPhysDShape(Trans, dshape);
            for (int n = 0; n < N; n++) {
                for (int l = 0; l < vdim; l++) {
                    auto [i, j] = this->IndexMap[l];
                    elmat(m+l*M,n+i*N) = dshape(n,j);
                }
            }
        }
    }
    else
    {
        for (int m = 0; m < M; m++)
        {
            const auto &ip  = e_fe.GetNodes().IntPoint(m);    
            Trans.SetIntPoint(&ip);
            u_fe.CalcPhysDShape(Trans, dshape);
            for (int n = 0; n < N; n++) {
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++)
                        elmat(m+i*M+j*dim*M,n+i*N) = dshape(n,j);
                }
            }
        }
    }
}


void StrainInterpolator::AssembleElementMatrix2(const FiniteElement &u_fe,
                                                const FiniteElement &e_fe,
                                                ElementTransformation &Trans,
                                                DenseMatrix &elmat)
{
    int N = u_fe.GetDof(), M = e_fe.GetDof();
 
    DenseMatrix dshape(N, dim);

    if (mode == VecMode::REDUCED) {
        elmat.SetSize(vdim*M, dim*N);
    } else {
        elmat.SetSize(dim*dim*M, dim*N); }
    elmat = 0.0;

    constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);
    if (mode == VecMode::REDUCED)
    {
        for (int m = 0; m < M; m++)
        {
            const auto &ip  = e_fe.GetNodes().IntPoint(m);    
            Trans.SetIntPoint(&ip);
            u_fe.CalcPhysDShape(Trans, dshape);
            for (int n = 0; n < N; n++) {
                for (int l = 0; l < vdim; l++) {
                    auto [i, j] = this->IndexMap[l];
                    elmat(m+l*M,n+i*N) += dshape(n,j);
                    elmat(m+l*M,n+j*N) += dshape(n,i);
                }
            }
        }
    }
    else 
    {
        for (int m = 0; m < M; m++)
        {
            const auto &ip  = e_fe.GetNodes().IntPoint(m);    
            Trans.SetIntPoint(&ip);
            u_fe.CalcPhysDShape(Trans, dshape);
            for (int n = 0; n < N; n++) {
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                    elmat(m+i*M+j*dim*M,n+i*N) += dshape(n,j);
                    elmat(m+i*M+j*dim*M,n+j*N) += dshape(n,i);
                    }
                }
            }

        }
    }
    elmat *= half;
}


void DevStrainInterpolator::AssembleElementMatrix2(const FiniteElement &u_fe,
                                                   const FiniteElement &e_fe,
                                                   ElementTransformation &Trans,
                                                   DenseMatrix &elmat)
{
    int N = u_fe.GetDof(), M = e_fe.GetDof();

    DenseMatrix dshape(N, dim);

    if (mode == VecMode::REDUCED) {
        elmat.SetSize(vdim*M, dim*N);
    } else {
        elmat.SetSize(dim*dim*M, dim*N); }
    elmat = 0.0;

    constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);
    auto two_thirds = 2.0 / dim;
    for (int m = 0; m < M; m++)
    {
        const auto &ip  = e_fe.GetNodes().IntPoint(m);    
        Trans.SetIntPoint(&ip);
        u_fe.CalcPhysDShape(Trans, dshape);
        if (mode == VecMode::REDUCED)
        {
            for (int n = 0; n < N; n++){
                for (int l = 0; l < vdim; l++){
                    auto [i, j] = this->IndexMap[l];
                    elmat(m+l*M,n+i*N) += dshape(n,j);
                    elmat(m+l*M,n+j*N) += dshape(n,i);
                    if (i == j){
                        for (int k = 0; k < dim; k++){
                            elmat(m+l*M,n+k*N) -= two_thirds * dshape(n,k);
                        }
                    }
                }
            }
        }
        else
        {
            for (int n = 0; n < N; n++){
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        elmat(m+i*M+j*dim*M,n+i*N) += dshape(n,j);
                        elmat(m+i*M+j*dim*M,n+j*N) += dshape(n,i);
                        if (i == j){
                            for (int k = 0; k < dim; k++)
                                elmat(m+i*M+j*dim*M,n+k*N) -= two_thirds * dshape(n,k);
                        }
                    }
                }
            }

        }
    }
    elmat *= half;
}


//Integrators
void ViscoelasticForcing::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                                 const FiniteElement &test_fe,
                                                 ElementTransformation &Trans,
                                                 DenseMatrix &elmat)
{
    int N = trial_fe.GetDof(), M = test_fe.GetDof();

    DenseMatrix dshape_test(M, dim);
    Vector shape_trial(N);

    DenseMatrix elmat_full(dim*M, vdim_full*N);
    elmat.SetSize(dim*M, vdim*N);
    elmat = 0.0; elmat_full = 0.0;
    real_t third = 1.0 / dim;

    //const IntegrationRule *ir = GetIntegrationRule(trial_fe, test_fe, Trans);
    //if (ir == NULL)
    //{
    int ir_order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() + 1;
    const IntegrationRule *ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
    //}
/*
    const IntegrationRule *ir = NULL;
    int ir_order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
    ir = &IntRules.Get(trial_fe.GetGeomType(), ir_order);
*/
    for (int q = 0; q < ir->GetNPoints(); q++)
    {
        const IntegrationPoint &ip = ir->IntPoint(q);
        Trans.SetIntPoint(&ip);
        real_t val = Trans.Weight() * ip.weight * mu.Eval(Trans, ip);
        test_fe.CalcPhysDShape(Trans, dshape_test);
        trial_fe.CalcShape(ip, shape_trial);

        std::vector<DenseMatrix> pelmats(dim);
        for (int l = 0; l < dim; l++) {
            Vector dshape_test_col(dshape_test.GetColumn(l), M);
            pelmats[l].SetSize(M, N);
            MultVWt(dshape_test_col, shape_trial, pelmats[l]);
        }
        for (int k = 0; k < dim; k++){
            for (int j = 0; j < dim; j++){
                for (int i = 0; i < dim; i++){
                    if (i == k){
                        elmat_full.AddMatrix(val, pelmats[j], k * M, i * N + j * dim * N);
                    }
                    if (j == k){
                        elmat_full.AddMatrix(val, pelmats[i], k * M, i * N + j * dim * N);
                    }
                    if (i == j){
                        elmat_full.AddMatrix(- val * 2.0 / 3.0, pelmats[k], k * M, i * N + j * dim * N);
                    } 
                }
            }
        }       
    }

   OperatorContractionTracefree(elmat_full, N, dim, elmat);
}


void AdvectionIntegrator::AssembleElementMatrix2(
    const FiniteElement &trial_fe, const FiniteElement &test_fe,
    ElementTransformation &Trans, DenseMatrix &elmat) 
{
    auto dim = Trans.GetSpaceDim();
    auto Nn = trial_fe.GetDof();
    auto Nm = test_fe.GetDof();

    elmat.SetSize(Nm, dim * Nn);
    elmat = 0.0;

#ifdef MFEM_THREAD_SAFE
    Vector trial_shape, qv;
    DenseMatrix test_dshape, pelmats, qm, tm;
#endif
    trial_shape.SetSize(Nn);
    test_dshape.SetSize(Nm, dim);
    pelmats.SetSize(Nm, Nn);

    if (QM) {
        qm.SetSize(dim);
        tm.SetSize(Nm, dim);
    } else if (QV) {
        qv.SetSize(dim);
    }

    const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

    for (auto i = 0; i < ir->GetNPoints(); i++) {
        const auto& ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        auto w = Trans.Weight() * ip.weight;

        trial_fe.CalcShape(ip, trial_shape);
        test_fe.CalcPhysDShape(Trans, test_dshape);

        if (QM) {//
            QM->Eval(qm, Trans, ip);
            MultABt(test_dshape, qm, tm);
            for (auto j = 0; j < dim; j++) {
                auto tm_column = Vector(tm.GetColumn(j), Nm);
                MultVWt(tm_column, trial_shape, pelmats);
                elmat.AddMatrix(w, pelmats, j * Nm, 0);
            }
        } else if (QV) {//
            QV->Eval(qv, Trans, ip);
            qv *= w;
            for (auto j = 0; j < dim; j++) {
                auto test_dshape_column = Vector(test_dshape.GetColumn(j), Nm);
                MultVWt(test_dshape_column, trial_shape, pelmats);
                elmat.AddMatrix(qv(j), pelmats, j * Nm, 0);
            }
        } else {
            if (Q) {
                w *= Q->Eval(Trans, ip);
            }
            for (auto j = 0; j < dim; j++) {
                auto test_dshape_column = Vector(test_dshape.GetColumn(j), Nm);
                //gshape.GetColumnReference(d, d_col);
                MultVWt(test_dshape_column, trial_shape, pelmats);
                elmat.AddMatrix(w, pelmats, 0, j * Nn);
            }
        }
    }
}


void ProjectionGradientIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat) 
{
    auto dim = Trans.GetSpaceDim();
    auto N = el.GetDof();

    elmat.SetSize(dim * N, dim * N);
    elmat = 0.0;

#ifdef MFEM_THREAD_SAFE
    Vector shape, g;
    DenseMatrix dshape, pelmat_l, pelmat_r, dg;
#endif
    shape.SetSize(N);
    dshape.SetSize(N, dim);
    pelmat_l.SetSize(N, N);
    pelmat_r.SetSize(N, N);
    g.SetSize(dim);
    dg.SetSize(dim, dim);

    const auto* ir = GetIntegrationRule(el, Trans);
    if (ir == NULL)
    {
        int order = order = 2 * el.GetOrder() + Trans.OrderW() - 1; //
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (auto i = 0; i < ir->GetNPoints(); i++) {
        const auto& ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        auto w = Trans.Weight() * ip.weight;

        el.CalcShape(ip, shape);
        el.CalcPhysDShape(Trans, dshape);

        QV->Eval(g, Trans, ip);
        QM->Eval(dg, Trans, ip);
        w *= Q->Eval(Trans, ip); 
        MultVWt(shape, shape, pelmat_r);
        for (auto k = 0; k < dim; k++) {
            auto dshape_column = Vector(dshape.GetColumn(k), N);
            MultVWt(shape, dshape_column, pelmat_l);
            for (auto l = 0; l < dim; l++) {
                real_t w_l = g(l) * w; 
                elmat.AddMatrix(w_l, pelmat_l, k * N, l * N);
                real_t w_r = dg(l, k) * w;
                elmat.AddMatrix(w_r, pelmat_r, k * N, l * N);
            }
        }
    }
}


void AdvectionProjectionIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    auto dim = Trans.GetSpaceDim();
    auto N = el.GetDof();

    elmat.SetSize(dim * N, dim * N);
    elmat = 0.0;

#ifdef MFEM_THREAD_SAFE
    Vector shape, g;
    DenseMatrix dshape, pelmat_l, pelmat_r, dg;
#endif
    shape.SetSize(N);
    dshape.SetSize(N, dim);
    pelmat_l.SetSize(N);
    pelmat_r.SetSize(N);
    g.SetSize(dim);
    dg.SetSize(dim, dim);

    const auto* ir = GetIntegrationRule(el, Trans);
    if (ir == NULL)
    {
        int order = order = 2 * el.GetOrder() + Trans.OrderW() - 1; //
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (auto i = 0; i < ir->GetNPoints(); i++) {
        const auto& ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        auto w = Trans.Weight() * ip.weight;

        el.CalcShape(ip, shape);
        el.CalcPhysDShape(Trans, dshape);

        QV->Eval(g, Trans, ip);
        QM->Eval(dg, Trans, ip);
        w *= Q->Eval(Trans, ip); 
        MultVWt(shape, shape, pelmat_r);
        for (auto k = 0; k < dim; k++) {
            for (auto l = 0; l < dim; l++) {
                auto dshape_column = Vector(dshape.GetColumn(l), N);
                MultVWt(dshape_column, shape, pelmat_l);
                real_t w_l = g(k) * w; 
                elmat.AddMatrix(w_l, pelmat_l, k * N, l * N);
                real_t w_r = dg(k, l) * w;
                elmat.AddMatrix(w_r, pelmat_r, k * N, l * N);
            }
        }
    }
}


void DivergenceVectorIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat) 
{
    auto dim = Trans.GetSpaceDim();
    auto N = el.GetDof();

    elmat.SetSize(dim * N);
    elmat = 0.0;

#ifdef MFEM_THREAD_SAFE
    Vector shape, g;
    DenseMatrix dshape, pelmat;
#endif
    dshape.SetSize(N, dim);
    shape.SetSize(N);
    pelmat.SetSize(N);
    g.SetSize(dim);

    const auto* ir = GetIntegrationRule(el, Trans);
    if (ir == NULL)
    {
        int order = order = 2 * el.GetOrder() + Trans.OrderW() - 1; //
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (auto i = 0; i < ir->GetNPoints(); i++) {
        const auto& ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        auto w = Trans.Weight() * ip.weight;

        el.CalcShape(ip, shape);
        el.CalcPhysDShape(Trans, dshape);

        QV->Eval(g, Trans, ip);
        w *= Q->Eval(Trans, ip); 
        for (auto k = 0; k < dim; k++) {
            for (auto l = 0; l < dim; l++) {
                auto dshape_column = Vector(dshape.GetColumn(l), N);
                MultVWt(shape, dshape_column, pelmat);
                real_t _w = w * g(k);
                elmat.AddMatrix(_w, pelmat, k * N, l * N);
            }
        }
    }
}


void ProjectionDivergenceIntegrator::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat) 
{
    auto dim = Trans.GetSpaceDim();
    auto N = el.GetDof();

    elmat.SetSize(dim * N);
    elmat = 0.0;

#ifdef MFEM_THREAD_SAFE
    Vector shape, g;
    DenseMatrix dshape, pelmat;
#endif
    dshape.SetSize(N, dim);
    shape.SetSize(N);
    pelmat.SetSize(N);
    g.SetSize(dim);

    const auto* ir = GetIntegrationRule(el, Trans);
    if (ir == NULL)
    {
        int order = order = 2 * el.GetOrder() + Trans.OrderW() - 1; //
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (auto i = 0; i < ir->GetNPoints(); i++) {
        const auto& ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        auto w = Trans.Weight() * ip.weight;

                el.CalcShape(ip, shape);
        el.CalcPhysDShape(Trans, dshape);

        QV->Eval(g, Trans, ip);
        w *= Q->Eval(Trans, ip); 
        for (auto k = 0; k < dim; k++) {
            auto dshape_column = Vector(dshape.GetColumn(k), N);
            MultVWt(dshape_column, shape, pelmat);
            for (auto l = 0; l < dim; l++) {
                real_t _w = w * g(l);
                elmat.AddMatrix(_w, pelmat, k * N, l * N);
            }
        }
    }
}


/*
const IntegrationRule& ProjDivIntegrator::GetRule(
    const FiniteElement &trial_fe, const FiniteElement &test_fe,
    const ElementTransformation &Trans) {
    const auto order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1;
    return IntRules.Get(trial_fe.GetGeomType(), order);
}

void ProjDivIntegrator::AssembleElementMatrix2(
    const FiniteElement &trial_fe, const FiniteElement &test_fe,
    ElementTransformation &Trans, DenseMatrix &elmat) 
{
    auto dim = Trans.GetSpaceDim();
    auto Nn = trial_fe.GetDof();
    auto Nm = test_fe.GetDof();

    elmat.SetSize(dim * Nm, dim * Nn);
    elmat = 0.0;

#ifdef MFEM_THREAD_SAFE
    Vector trial_shape, g;
    DenseMatrix test_dshape, pelmats;
#endif
    test_dshape.SetSize(Nm, dim);
    trial_shape.SetSize(Nn);
    pelmats.SetSize(Nm, Nn);
    g.SetSize(dim);

    const auto* ir = GetIntegrationRule(trial_fe, test_fe, Trans);

    for (auto i = 0; i < ir->GetNPoints(); i++) {
        const auto& ip = ir->IntPoint(i);
        Trans.SetIntPoint(&ip);
        auto w = Trans.Weight() * ip.weight;

        test_fe.CalcPhysDShape(Trans, test_dshape);
        trial_fe.CalcShape(ip, trial_shape);

        QV->Eval(g, Trans, ip);
        w *= Q->Eval(Trans, ip); 
        for (auto k = 0; k < dim; k++) {
            auto test_dshape_column = Vector(test_dshape.GetColumn(k), Nm);
            MultVWt(test_dshape_column, trial_shape, pelmats);
            for (auto l = 0; l < dim; l++) {
                w *= g(l);
                elmat.AddMatrix(w, pelmats, k * Nm, l * Nn);
            }
        }
    }
}
*/




void ViscoelasticRHSIntegrator::AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvec)
{
    int dof = el.GetDof();
    int dim = el.GetDim();
    elvec.SetSize(dof * dim);
    elvec = 0.0;

    const IntegrationRule *ir = GetIntegrationRule(el, Tr);
    if (ir == NULL)
    {
        ir = &IntRules.Get(el.GetGeomType(), el.GetOrder() * 2 + Tr.OrderW());
    }
    Vector m_vec;
    DenseMatrix m_tensor(dim, dim);
    DenseMatrix dshape(dof, dim);

    for (int q = 0; q < ir->GetNPoints(); q++)
    {
        const IntegrationPoint &ip = ir->IntPoint(q);
        Tr.SetIntPoint(&ip);
        m.Eval(m_vec, Tr, ip);
        if (dim == 2){
            m_tensor(0, 0) = m_vec(0); m_tensor(1, 0) = m_vec(1); m_tensor(0, 1) = m_vec(1); m_tensor(1, 1) = -m_vec(0);
        }
        else {
            m_tensor(0, 0) = m_vec(0); m_tensor(1, 0) = m_vec(1); m_tensor(2, 0) = m_vec(2);
            m_tensor(0, 1) = m_vec(1); m_tensor(1, 1) = m_vec(3); m_tensor(2, 1) = m_vec(4);
            m_tensor(0, 2) = m_vec(2); m_tensor(1, 2) = m_vec(4); m_tensor(2, 2) = -m_vec(0)-m_vec(3);
        }

        real_t val = Tr.Weight() * ip.weight * mu.Eval(Tr, ip);

        el.CalcPhysDShape(Tr, dshape);

        for (int m = 0; m < dof; m++)
        {
            for (int k = 0; k < dim; k++)
            {
                real_t increment = 0.0; 
                for (int s = 0; s < dim; s++)
                {
                   increment += 2 * m_tensor(k, s) * dshape(m, s);
                }
                //increment -= 2.0/3.0 * m_tensor.Trace() * dshape(m, k);
                elvec(m + k * dof) += val * increment;
            }
        }
    }
}


GradientVectorGridFunctionCoefficient::GradientVectorGridFunctionCoefficient(
    const GridFunction *gf) : MatrixCoefficient((gf) ? gf->VectorDim() : 0,
                                                (gf) ? gf->FESpace()->GetMesh()->SpaceDimension() : 0), GridFunc(gf)
{ }

void GradientVectorGridFunctionCoefficient::SetGridFunction(
    const GridFunction *gf)
{
    GridFunc = gf;
    height = (gf) ? gf->VectorDim() : 0;
    width = (gf) ? gf->FESpace()->GetMesh()->SpaceDimension() : 0;
}

void GradientVectorGridFunctionCoefficient::Eval(
    DenseMatrix &M,
    ElementTransformation &T,
    const IntegrationPoint &ip)
{
    Mesh *gf_mesh = GridFunc->FESpace()->GetMesh();
    //if (T.mesh->GetNE() == gf_mesh->GetNE())
    {
        GridFunc->GetVectorGradient(T, M);
    }/*
    else
    {
        IntegrationPoint coarse_ip;
        ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
        GridFunc->GetVectorGradient(*coarse_T, M);
    }*/
}
/*
void GradientVectorGridFunctionCoefficient::Eval(
    mfem::DenseTensor &T,
    mfem::ElementTransformation &Tr,
    const mfem::IntegrationRule &ir)
{
    if (Tr.mesh == GridFunc->FESpace()->GetMesh())
    {
        GridFunc->GetVectorGradients(Tr, ir, T);
    }
    else
    {
        mfem::MatrixCoefficient::Eval(T, Tr, ir);
    }
}
*/


//Solver-related
void BaileySolver::Init(TimeDependentOperator &f_)
{
    ODESolver::Init(f_);
    dxdt.SetSize(f->Width(), mem_type);
    d_vec_old.SetSize(f->Width());
    d_vec_new.SetSize(f->Width());
    //d_vec_old = 0.0;
}

void BaileySolver::Step(Vector &x, real_t &t, real_t &dt)
{
    auto &op = *static_cast<VeOperator *>(f);
    f->SetTime(t);
    op.Mult(x, dxdt);
    //op.GetDev().GetTrueDofs(d_vec_new);
    Vector tau_vec;
    op.GetTau().GetTrueDofs(tau_vec);
/*
    if (t == 0.0)  // First time step: use Forward Euler
    {
        for (int i = 0; i < x.Size(); i++)
        {
            x[i] += dt * dxdt[i];
        }
    }
    else */ 
    {
        for (int i = 0; i < x.Size(); i++)
        {
            real_t tau = tau_vec[i % tau_vec.Size()];
            real_t decay = exp(-dt / tau);

            x[i] = x[i] * decay + (1.0 - decay) * (x[i] + tau * dxdt[i]);
        }
    }

    d_vec_old = d_vec_new;

    t += dt;
}



//Plotting
plot::plot(const GridFunction &sol, Mesh &msh)
    : solution(sol), mesh(msh) {}

plot::~plot()
{
    //delete[] data;
    std::cout << "Destroying plot object and releasing data." << std::endl;
}

parse::~parse()
{
    std::cout << "Destroying parse object." << std::endl;
}

interp::~interp()
{
    std::cout << "Destroying interp object." << std::endl;
}


void plot::EvaluateRadialSolution(int num_samples, double min_radius, double max_radius, 
        std::vector<double> &radii, std::vector<double> &values)
{
    double mesh_max_radius = 0.0;

    for (int i = 0; i < mesh.GetNV(); i++)
    {
        Vector node;
        //mesh.GetVertex(i, node);
        const double* vertex_coords = mesh.GetVertex(i);
        node.SetSize(mesh.Dimension());
        for (int d = 0; d < mesh.Dimension(); d++)
        {
            node(d) = vertex_coords[d];
        }

        double radius = node.Norml2();
        if (radius > mesh_max_radius)
            mesh_max_radius = radius;
    }
    max_radius = std::min(max_radius, mesh_max_radius);


    DenseMatrix points(mesh.Dimension(), num_samples);
    for (int i = 0; i < num_samples; i++)
    {
        double radius = min_radius + i * (max_radius - min_radius) / num_samples;
        points(0, i) = radius;
        if (mesh.Dimension() > 1) points(1, i) = 0.0;
        if (mesh.Dimension() > 2) points(2, i) = 0.0; 
    }


    Array<int> elem_ids(num_samples);
    Array<IntegrationPoint> ips(num_samples);

    mesh.FindPoints(points, elem_ids, ips);

    for (int i = 0; i < num_samples; i++)
    {
        double value = 0.0;
        int elem_id = elem_ids[i];
        if (elem_id >= 0) // Check if the point was found
        {
            ElementTransformation *trans = mesh.GetElementTransformation(elem_id);

            value = solution.GetValue(*trans, ips[i]);
            std::cout << "Point (" << points(0, i) << ", "
                << (mesh.Dimension() > 1 ? points(1, i) : 0.0) << ") "
                << "has value " << value << " in element " << elem_id << ".\n";
        }
        else
        {
            std::cerr << "Warning: Point (" << points(0, i) << ", "
                << (mesh.Dimension() > 1 ? points(1, i) : 0.0) << ") "
                << "is outside the mesh.\n";
        }

        radii.push_back(points(0, i));
        values.push_back(value);
    }
}

void plot::SaveRadialSolution(const std::string &filename, int num_samples, double min_radius, double max_radius)
{
    std::vector<double> radii;
    std::vector<double> values;
    EvaluateRadialSolution(num_samples, min_radius, max_radius, radii, values);

    std::ofstream ofs(filename);
    //if (!outfile.is_open()) {
    //    std::cerr << "Error: Could not open the file " << filename << " for writing.\n";
    //    return;
    //}

    ofs << filename<<"\n";
    for (size_t i = 0; i < radii.size(); i++)
    {
        ofs << radii[i] << " " << values[i] << "\n";
    }
    ofs.close();
}



std::vector<std::vector<double>> parse::properties_1d(const std::string &filename)
{
    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::vector<std::vector<double>> properties;
    std::string line;

    int num_columns = 0; // Local variable to track the number of columns

    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);

    // Parse the file
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;

        // Read all columns in the row
        while (iss >> value)
        {
            row.push_back(value);
        }

        // Determine the number of columns from the first valid row
        if (properties.empty() && !row.empty())
        {
            num_columns = row.size();
        }

        // Ensure all rows have the same number of columns
        if (!row.empty() && row.size() != num_columns)
        {
            throw std::runtime_error("Inconsistent number of columns in file: " + filename);
        }

        if (!row.empty())
        {
            properties.push_back(row);
        }
    }

    infile.close();

    return properties;
}


    mfem::Array<mfem::FunctionCoefficient *>
interp::PWCoef_1D(const std::vector<std::pair<double, double>> &radius_property,
        int num_attributes,
        const std::string &method)
{
    if (radius_property.empty())
    {
        throw std::runtime_error("Radius-property array is empty.");
    }

    // Define the interpolation function
    std::function<double(double)> interpolate;

    if (method == "linear")
    {
        interpolate = [radius_property](double radius) -> double {
            if (radius > radius_property.back().first)
            {
                return 0.0; // Return zero outside the range
            }

            for (size_t i = 1; i < radius_property.size(); ++i)
            {
                if (radius <= radius_property[i].first)
                {
                    double r1 = radius_property[i - 1].first;
                    double r2 = radius_property[i].first;
                    double v1 = radius_property[i - 1].second;
                    double v2 = radius_property[i].second;

                    // Linear interpolation
                    return v1 + (v2 - v1) * (radius - r1) / (r2 - r1);
                }
            }
            return 0.0; 
        };
    }
    else if (method == "spline")
    {
        throw std::runtime_error("Spline interpolation is not implemented yet.");
    }
    else
    {
        throw std::runtime_error("Unsupported interpolation method: " + method);
    }

    // Create FunctionCoefficients for each attribute
    mfem::Array<mfem::FunctionCoefficient *> coefficients;
    for (int i = 1; i <= num_attributes; ++i)
    {
        // Create a FunctionCoefficient for this attribute
        mfem::FunctionCoefficient *fc = new mfem::FunctionCoefficient([interpolate](const mfem::Vector &x) -> double {
                double r = x.Norml2();
                return interpolate(r);
                });

        coefficients.Append(fc);
    }

    return coefficients;
}



//Legacy
void visualize(ostream &os, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name, bool init_vis)
{
    GridFunction *displaced_nodes = new GridFunction(deformed_nodes->FESpace());
    *displaced_nodes = *mesh->GetNodes();          // base geometry
    real_t fac = 5000;
    displaced_nodes->Add(fac, *deformed_nodes);

    int owns_nodes = 0;
    mesh->SwapNodes(displaced_nodes, owns_nodes);
    
    os << "solution\n" << *mesh << *field;

    if (init_vis)
    {
        os << "window_size 800 800\n";
        os << "window_title '" << field_name << "'\n";
        if (mesh->SpaceDimension() == 2)
        {
            os << "view 0 0\n"; 
            os << "keys jl\n";  
        }
        os << "keys cm\n";        
        os << "autoscale value\n";
        os << "pause\n";
    }
    mesh->SwapNodes(displaced_nodes, owns_nodes);

    os << flush;

    delete displaced_nodes;
}


VeOperator::VeOperator(FiniteElementSpace &fes_u_, FiniteElementSpace &fes_m_, FiniteElementSpace &fes_properties_, FiniteElementSpace &fes_w_, GridFunction &u_gf_, GridFunction &m_gf_, GridFunction &d_gf_, Coefficient &lamb_, Coefficient &mu_, Coefficient &tau_, Coefficient &loading_)   
    : TimeDependentOperator(fes_m_.GetTrueVSize(), (real_t) 0.0), fes_u(fes_u_), fes_m(fes_m_), fes_properties(fes_properties_), fes_w(fes_w_), u_gf(u_gf_), m_gf(m_gf_), d_gf(d_gf_), lamb_gf(&fes_properties), mu_gf(&fes_properties), tau_gf(&fes_properties), lamb(lamb_), mu(mu_), tau(tau_), loading(loading_), K(NULL), B(&fes_m, &fes_u), B2(&fes_u, &fes_m), current_dt(0.0), Dev(&fes_u_, &fes_m_), rigid_solver(&fes_u_) 
{
    tau_gf.ProjectCoefficient(tau);
    tau_gf.GetTrueDofs(tau_vec);

    Array<int> ess_bdr(fes_u.GetMesh()->bdr_attributes.Max());
    ess_bdr = 0;
    fes_u.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    K = new BilinearForm(&fes_u);
    K->AddDomainIntegrator(new ElasticityIntegrator(lamb, mu));
    K->Assemble();
    K->FormSystemMatrix(ess_tdof_list, Kmat);
    
    K_solver.iterative_mode = false;
    K_solver.SetRelTol(rel_tol);
    K_solver.SetAbsTol(0.0);
    K_solver.SetMaxIter(1000);
    K_solver.SetPrintLevel(0);
    K_solver.SetPreconditioner(K_prec);
    K_solver.SetOperator(Kmat);
    rigid_solver.SetSolver(K_solver);

    B.AddDomainIntegrator(new ViscoelasticForcing(mu));
    B.Assemble();

    B2.AddDomainIntegrator(new mfemElasticity::DomainTraceFreeSymmetricMatrixDeviatoricStrainIntegrator(mu));
    B2.Assemble();

    Dev.AddDomainInterpolator(new DevStrainInterpolator);
    Dev.Assemble();
}

void VeOperator::Mult(const Vector &m_vec, Vector &dm_dt_vec) const
{
    LinearForm *b = new LinearForm(&fes_u);
    //m_gf.SetFromTrueDofs(m_vec);
    //VectorGridFunctionCoefficient m_coeff(&m_gf);
    //b->AddDomainIntegrator(new ViscoelasticRHSIntegrator(mu, m_coeff));
    b->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(loading));
    b->Assemble();

    B.AddMult(m_vec, *b);

    Vector b1 (b->Size());
    Vector b2 (b->Size());
    B.Mult(m_vec, b1);
    B2.MultTranspose(m_vec, b2);
    cout<<B.Height()<<" "<<B2.Height()<<endl;
    cout<<B.Width()<<" "<<B2.Width()<<endl;
    cout<<"b1: "<<b1.Norml2()<<endl;
    cout<<"b2: "<<b2.Norml2()<<endl;

    K->FormLinearSystem(ess_tdof_list, u_gf, *b, Kmat, x_vec, b_vec);
    rigid_solver.Mult(*b, x_vec);
    K->RecoverFEMSolution(x_vec, *b, u_gf);

    Dev.Mult(u_gf, d_gf);
    d_gf.GetTrueDofs(d_vec);
    for (int i = 0; i < d_vec.Size(); i++)
    {
        dm_dt_vec[i] = (d_vec[i] - m_vec[i]) / tau_vec[i % tau_vec.Size()];
    }

    delete b;
}

void VeOperator::ImplicitSolve(const real_t dt, const Vector &m_vec, Vector &dm_dt_vec)
{
    current_dt = dt;
    this->Mult(m_vec, dm_dt_vec);
    Vector dm_dt_old;
    Vector res;
    int iter=0;
    do {
    cout<<"Iter: "<<++iter<<endl;
    dm_dt_old = dm_dt_vec;
    Vector m_est;
    //add(m_vec, current_dt, dm_dt_vec, m_est);
    m_est = dm_dt_vec; m_est *= current_dt; m_est += m_vec;
    this->Mult(m_est, dm_dt_vec);
    for (int i = 0; i < dm_dt_vec.Size(); i++)
    {
        dm_dt_vec[i] *= tau_vec[i % tau_vec.Size()];
        dm_dt_vec[i] /= tau_vec[i % tau_vec.Size()] + current_dt;
    }
    res = dm_dt_vec.Add(-1.0, dm_dt_old);
    //res = dm_dt_vec; res -= dm_dt_old;
    cout<<"Res: "<<res.Norml2()/dm_dt_old.Norml2()<<endl;
    } while (res.Norml2()/dm_dt_old.Norml2() > res_max);

}

void VeOperator::CalcStrainEnergyDensity(GridFunction &w_gf)
{
    StrainEnergyCoefficient w_coeff(u_gf, lamb, mu);
    w_gf.ProjectCoefficient(w_coeff);
}


VeOperator_beam::VeOperator_beam(FiniteElementSpace &fes_u_, FiniteElementSpace &fes_m_, FiniteElementSpace &fes_w_, 
                       Coefficient &lamb_, Coefficient &mu_, Coefficient &tau_, GridFunction &u_gf_, GridFunction &m_gf_, GridFunction &d_gf_)   
    : TimeDependentOperator(fes_m_.GetTrueVSize(), (real_t) 0.0), fes_u(fes_u_), fes_m(fes_m_), fes_w(fes_w_), u_gf(u_gf_), m_gf(m_gf_), d_gf(d_gf_), lamb(lamb_), mu(mu_), tau(tau_), K(NULL), current_dt(0.0), force(3), lamb_gf(&fes_w), mu_gf(&fes_w), tau_gf(&fes_w), Dev(&fes_u_, &fes_m_), loading(nullptr), fes_properties(fes_w) 
{
    tau_gf.ProjectCoefficient(tau);
    tau_gf.GetTrueDofs(tau_vec);

    Array<int> ess_bdr(fes_u.GetMesh()->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;
    fes_u.GetEssentialTrueDofs(ess_bdr, etl);
    
    int dim = fes_u.GetMesh()->Dimension();
    for (int i = 0; i < dim-1; i++)
    {
       force.Set(i, new ConstantCoefficient(0.0));
    }
    {
       Vector pull_force(fes_u.GetMesh()->bdr_attributes.Max());
       pull_force = 0.0;
       pull_force(1) = -1.0e-2;
       force.Set(dim-1, new PWConstCoefficient(pull_force));
    }
    create_boundary_integ = [this]() {
        return new VectorBoundaryLFIntegrator(force);
    };

    const real_t rel_tol = 1e-6;

    auto b = new LinearForm(&fes_u);
    VectorGridFunctionCoefficient m_func(&m_gf);
    b->AddDomainIntegrator(new ViscoelasticRHSIntegrator(mu, m_func));
    b->AddBoundaryIntegrator(create_boundary_integ());
    b->Assemble();
    
    K = new BilinearForm(&fes_u);
    K->AddDomainIntegrator(new ElasticityIntegrator(lamb, mu));
    K->Assemble();
    K->FormSystemMatrix(etl, Kmat);
    
    K_solver.iterative_mode = false;
    K_solver.SetRelTol(rel_tol);
    K_solver.SetAbsTol(0.0);
    K_solver.SetMaxIter(1000);
    K_solver.SetPrintLevel(0);
    K_solver.SetPreconditioner(K_prec);

    Dev.AddDomainInterpolator(new DevStrainInterpolator);
    Dev.Assemble();
    //Dev.Finalize();
}

void VeOperator_beam::Mult(const Vector &m_vec, Vector &dm_dt_vec) const
{
    auto b = new LinearForm(&fes_u);
    m_gf.SetFromTrueDofs(m_vec);
    VectorGridFunctionCoefficient m_func(&m_gf);
    b->AddDomainIntegrator(new ViscoelasticRHSIntegrator(mu, m_func));
    b->AddBoundaryIntegrator(create_boundary_integ());
    b->Assemble();
    
    Vector X, B;
    K->FormLinearSystem(etl, u_gf, *b, Kmat, X, B);
    K_solver.SetOperator(Kmat);
    K_solver.Mult(B, X);
    K->RecoverFEMSolution(X, *b, u_gf);

    //StrainCoefficient d_func(&u_gf, StrainCoefficient::DEVIATORIC);
    //d_gf.ProjectCoefficient(d_func);
    //d_gf.GetTrueDofs(d_vec);
    
    Dev.Mult(u_gf, d_gf);
    d_gf.GetTrueDofs(d_vec);

    //Vector d_vec2;
    //u_gf.GetTrueDofs(u_vec);
    //Dev.Mult(u_vec,d_vec2);
    //Cout<<"Error of dev. stress: "<<d_vec.ComputeL2Error(d_vec2)<<endl;

    for (int i = 0; i < d_vec.Size(); i++)
    {
        //d_vec[i] = 1;
        dm_dt_vec[i] = (d_vec[i] - m_vec[i]) / tau_vec[i % tau_vec.Size()];
    }

    delete b;
}

void VeOperator_beam::ImplicitSolve(const real_t dt,
                               const Vector &m, Vector &dm_dt)
{
    current_dt = dt;
    this->Mult(m, dm_dt);
    Vector dm_dt_old;
    Vector res;
    int iter=0;
    do {
    cout<<"Iter: "<<++iter<<endl;
    dm_dt_old = dm_dt;
    //add(m, current_dt, dm_dt, z);
    Vector m_est;
    m_est = dm_dt; m_est *= current_dt; m_est += m;
    //add(m, current_dt, dm_dt, m_est);
    //m_est = Add(1.0, m, current_dt, dm_dt);
    //z = Add(1.0, m, current_dt, dm_dt);
    this->Mult(m_est, dm_dt);

    for (int i = 0; i < dm_dt.Size(); i++)
    {
        dm_dt[i] *= tau_vec[i % tau_vec.Size()];
        dm_dt[i] /= tau_vec[i % tau_vec.Size()] + current_dt;
    }
    cout<<"dmdt: "<<dm_dt.Norml2()<<endl;
    cout<<"dmdt_old: "<<dm_dt_old.Norml2()<<endl;


    res = dm_dt;
    res -= dm_dt_old;
    cout<<"res: "<<res.Norml2()<<endl;
    } while (res.Norml2()>1e-8);

}


void BaileySolver_test::Init(TimeDependentOperator &f_)
{
    ODESolver::Init(f_);
    dxdt.SetSize(f->Width(), mem_type);
    d_vec_old.SetSize(f->Width());
    d_vec_new.SetSize(f->Width());
    //d_vec_old = 0.0;
}

void BaileySolver_test::Step(Vector &x, real_t &t, real_t &dt)
{
    f->SetTime(t);
    f->Mult(x, dxdt);
    {
        for (int i = 0; i < x.Size(); i++)
        {
            real_t decay = exp(-dt / tau);

            x[i] = x[i] * decay + (1.0 - decay) * (x[i] + tau * dxdt[i]);
        }
    }

    t += dt;
}



}

