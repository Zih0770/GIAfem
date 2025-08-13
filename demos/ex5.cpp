#include <mfem.hpp>
#include <giafem.hpp>
#include <mfemElasticity.hpp>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;

real_t rho_func(const Vector &coord);
real_t mu_func(const Vector &coord);
real_t lamb_func(const Vector &coord);
real_t loading_func(const Vector &coord);


int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/Earth_space.msh";
    const char *elasticity_model = "linear";
    real_t rel_tol = 1e-10;
    int order_u = 1;
    int lMax = 10;
    const char *device_config = "cpu";
    bool visualization = false;

    //Parsing
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
            "Mesh file to use.");
    args.AddOption(&elasticity_model, "-em", "--elasticity-model",
            "Elasticity model to use: linear, neo-hookean, etc.");
    args.AddOption(&rel_tol, "-rt", "--rel-tol",
            "Relative tolerance for linear solving.");
    args.AddOption(&order_u, "-o", "--order",
            "Order (degree) of the finite elements.");
    args.AddOption(&lMax, "-l", "--lMax", "Truncation degree for the DtN map.");
    args.AddOption(&device_config, "-d", "--device",
            "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
            "--no-visualization",
            "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    Device device(device_config);
    device.Print();

    //Mesh
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    Array<int> attr_cond = mesh->attributes;
    attr_cond.DeleteLast();

    SubMesh mesh_cond(SubMesh::CreateFromDomain(*mesh, attr_cond));

    //FE Space
    int order_phi = order_u; int order_dphi = order_phi - 1; int order_prop = order_u; 
    H1_FECollection fec_u(order_u, dim), fec_phi(order_phi, dim);
    L2_FECollection fec_dphi(order_dphi, dim), fec_prop(order_prop, dim);
    FiniteElementSpace fes_phi(mesh, &fec_phi), fes_phi_cond(&mesh_cond, &fec_phi), fes_dphi(mesh, &fec_dphi), fes_dphi_cond(&mesh_cond, &fec_dphi);
    FiniteElementSpace fes_prop(&mesh_cond, &fec_prop);
    FiniteElementSpace fes_u(&mesh_cond, &fec_u, dim);
    int u_size = fes_u.GetVSize();
    int phi_size = fes_phi.GetVSize();
    cout << "Number of u-unknowns: " << u_size << endl;
    cout << "Number of phi-unknowns: " << phi_size << endl;
    GridFunction u_gf(&fes_u); GridFunction phi_gf(&fes_phi); GridFunction phi_gf_cond(&fes_phi_cond); 
    GridFunction phi0_gf(&fes_phi); GridFunction phi0_gf_cond(&fes_phi_cond); GridFunction dphi0_gf(&fes_dphi); 
    GridFunction dphi0_gf_cond(&fes_dphi_cond);
    GridFunction rho_gf(&fes_prop), lamb_gf(&fes_prop), mu_gf(&fes_prop);
    u_gf = 0.0; phi_gf = 0.0; phi_gf_cond = 0.0; phi0_gf = 0.0; phi0_gf_cond = 0.0; dphi0_gf = 0.0; dphi0_gf_cond = 0.0;
    FunctionCoefficient rho_coeff(rho_func);
    FunctionCoefficient mu_coeff(mu_func);
    FunctionCoefficient lamb_coeff(lamb_func);
    FunctionCoefficient loading_coeff(loading_func);
    rho_gf.ProjectCoefficient(rho_coeff);
    lamb_gf.ProjectCoefficient(lamb_coeff);
    mu_gf.ProjectCoefficient(mu_coeff);

    Array<int> ess_tdof_list;

    Array<int> Earth_body_marker;
    Earth_body_marker = Array<int>(mesh->attributes.Size());
    Earth_body_marker = 1;
    Earth_body_marker[mesh->attributes.Size() - 1] = 0;

    Array<int> bdr_marker;
    int size = mesh->bdr_attributes.Size();
    bdr_marker = Array<int>(size);
    bdr_marker = 0;
    bdr_marker[size - 2] = 1;

    Array<int> bdr_marker_cond;
    int size_cond = mesh_cond.bdr_attributes.Size();
    bdr_marker_cond = Array<int>(size_cond);
    bdr_marker_cond = 0;
    bdr_marker_cond[size - 1] = 1;

    //Compute the equilibrium state
    auto DtN = DirichletToNeumannOperator(&fes_phi, lMax);

    ProductCoefficient rhs_coeff(- 4.0 * M_PI * Constants::G, rho_coeff);
    LinearForm b0(&fes_phi);
    b0.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
    b0.Assemble();

    BilinearForm a0(&fes_phi);
    auto one = ConstantCoefficient(1.0);
    a0.AddDomainIntegrator(new DiffusionIntegrator(one));
    a0.Assemble();

    OperatorPtr A0;
    Vector B0, Phi0;

    a0.FormLinearSystem(ess_tdof_list, phi0_gf, b0, A0, Phi0, B0);
    cout << "Size of linear system: " << A0->Height() << endl;

    auto S = SumOperator(A0.Ptr(), 1.0, &DtN, 1.0, false, false);

    GSSmoother M((SparseMatrix &)(*A0));

    auto solver0 = CGSolver();
    solver0.SetOperator(S);
    solver0.SetPreconditioner(M);
    solver0.SetRelTol(rel_tol);
    solver0.SetMaxIter(2000);
    solver0.SetPrintLevel(0);
    solver0.Mult(B0, Phi0);

    a0.RecoverFEMSolution(Phi0, b0, phi0_gf);

    DiscreteLinearOperator Grad(&fes_phi_cond, &fes_dphi_cond);
    Grad.AddDomainInterpolator(new GradientInterpolator);
    Grad.Assemble();

    GridFunctionCoefficient phi0_coeff(&phi0_gf);
    mesh_cond.Transfer(phi0_gf, phi0_gf_cond);
    Grad.Mult(phi0_gf_cond, dphi0_gf_cond);

    GradientGridFunctionCoefficient dphi0_coeff(&phi0_gf);
    VectorGridFunctionCoefficient dphi0_cond_coeff(&dphi0_gf_cond);
    GradientVectorGridFunctionCoefficient ddphi0_cond_coeff(&dphi0_gf_cond);
    ScalarVectorProductCoefficient dphi0_sig_cond_coeff(loading_coeff, dphi0_cond_coeff); //
                                                                                          
    cout<<"Equilibrium state computed."<<endl;

    if (!visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << *mesh << phi0_gf << flush;
    }

    //Coupled problem
    Vector U, Phi, B1, B2;
    u_gf.GetTrueDofs(U);
    phi_gf.GetTrueDofs(Phi);
        
    LinearForm *b1(new LinearForm(&fes_u)); //
    b1->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(dphi0_sig_cond_coeff), bdr_marker_cond); //! by luck
    b1->Assemble();
    
    LinearForm *b2(new LinearForm(&fes_phi));
    b2->AddBoundaryIntegrator(new BoundaryLFIntegrator(loading_coeff), bdr_marker);
    b2->Assemble();
    
    BilinearForm *a11(new BilinearForm(&fes_u));
    BilinearForm *a22(new BilinearForm(&fes_phi));
    ExtTrialMixedBilinearForm *a12(new ExtTrialMixedBilinearForm(&fes_phi, &fes_u, &fes_phi_cond, &mesh_cond));
    ExtTestMixedBilinearForm *a21(new ExtTestMixedBilinearForm(&fes_u, &fes_phi, &fes_phi_cond, &mesh_cond));
    
    ConstantCoefficient c0(1.0 / (4.0 * M_PI * Constants::G));
    ProductCoefficient half_rho_coeff(0.5, rho_coeff);
    ProductCoefficient minus_half_rho_coeff(-0.5, rho_coeff);
    ProductCoefficient minus_rho_coeff(-1.0, rho_coeff);
    
    auto a11_integ_0 = ElasticityIntegrator(lamb_coeff, mu_coeff);
    auto a11_integ_1 = AdvectionProjectionIntegrator(half_rho_coeff, dphi0_cond_coeff, ddphi0_cond_coeff);
    //auto a11_integ_1 = mfemElasticity::DomainVectorGradVectorIntegrator(dphi0_cond_coeff, half_rho_coeff);
    auto a11_integ_2 = ProjectionDivergenceIntegrator(minus_half_rho_coeff, dphi0_cond_coeff);
    auto a11_integ_1_t = TransposeIntegrator(&a11_integ_1);
    auto a11_integ_2_t = TransposeIntegrator(&a11_integ_2);
    a11->AddDomainIntegrator(&a11_integ_0);
    a11->AddDomainIntegrator(&a11_integ_1);
    a11->AddDomainIntegrator(&a11_integ_2);
    a11->AddDomainIntegrator(&a11_integ_1_t);
    a11->AddDomainIntegrator(&a11_integ_2_t);
    //a11->AddDomainIntegrator(new ElasticityIntegrator(lamb_coeff, mu_coeff));
    //a11->AddDomainIntegrator(new ProjectionGradientIntegrator(half_rho_coeff, dphi0_cond_coeff, ddphi0_cond_coeff));
    //a11->AddDomainIntegrator(new AdvectionProjectionIntegrator(half_rho_coeff, dphi0_cond_coeff, ddphi0_cond_coeff));
    //a11->AddDomainIntegrator(new DivergenceVectorIntegrator(minus_half_rho_coeff, dphi0_cond_coeff));
    //a11->AddDomainIntegrator(new ProjectionDivergenceIntegrator(minus_half_rho_coeff, dphi0_cond_coeff));
    a11->Assemble();
    a11->Finalize();
    
    a22->AddDomainIntegrator(new DiffusionIntegrator(c0));
    a22->Assemble();
    a22->Finalize();
    
    a12->AddDomainIntegrator(new GradientIntegrator(rho_coeff));
    a12->Assemble();
    a12->Finalize();
    
    SparseMatrix &A11(a11->SpMat());
    SparseMatrix &A12(a12->SpMat());
    SparseMatrix &A22_0(a22->SpMat());
    TransposeOperator A21(&A12);
    auto A22 = SumOperator(&A22_0, 1.0, &DtN, 1.0 / (4.0 * M_PI * Constants::G), false, false);

    cout<<"Max Norm of A11: "<<A11.MaxNorm()<<", A12: "<<A12.MaxNorm()<<", A22_0: "<<A22_0.MaxNorm()<<endl;
    cout<<"Asymmetry tests: A11: "<<A11.IsSymmetric()<<", A12: "<<A12.IsSymmetric()
        <<", A22_0: "<<A22_0.IsSymmetric()<<endl;
    
    GSSmoother prec11(A11);
    GSSmoother prec22(A22_0);
    
    MINRESSolver solver1;
    //CGSolver solver1;
    solver1.SetRelTol(rel_tol);
    solver1.SetMaxIter(3000);
    solver1.SetOperator(A11);
    solver1.SetPreconditioner(prec11);
    solver1.SetPrintLevel(0);

    RigidBodySolver rigid_solver(&fes_u);
    rigid_solver.SetSolver(solver1);

    CGSolver solver2;
    solver2.SetRelTol(rel_tol);
    solver2.SetMaxIter(3000);
    solver2.SetOperator(A22);
    solver2.SetPreconditioner(prec22);
    solver2.SetPrintLevel(0);

    int max_iter = 1000;
    int iter = 0;
    real_t rel_tol_coup = 1e-6;
    LinearForm b1_ext(&fes_u);
    LinearForm b2_ext(&fes_phi);
    Vector Phi_temp(Phi.Size()), Phi_diff(Phi.Size());
    Phi_temp = 0.0; Phi_diff = 0.0;
    for (int i = 0; i < max_iter; i++)
    {
        iter++;
        cout<<"Iteration "<<iter<<":"<<endl;
        //b1_ext.Update(&fes_u, *b1, 0);
        //b2_ext.Update(&fes_phi, *b2, 0);
        b1_ext = *b1;
        b2_ext = *b2;
        A12.AddMult(Phi, b1_ext, -1.0);
        rigid_solver.Mult(b1_ext, U);

        A21.AddMult(U, b2_ext, -1.0);
        solver2.Mult(b2_ext, Phi_temp);
        Phi_diff = Phi_temp; Phi_diff -= Phi;

        real_t res = Phi_diff.Norml2() / Phi_temp.Norml2();
        Phi = Phi_temp;
        cout<<"Residual = "<<res<<" after iteration "<<iter<<"."<<endl;
        if (res < rel_tol_coup)
        {
            cout<<"Converged at iteration "<<iter<<"."<<endl;
            break;
        }
    }
    if (iter == 1000)
    {
        cout<<"Not Converged after "<<iter<<" iterations."<<endl;
    }

    u_gf.SetFromTrueDofs(U);
    phi_gf.SetFromTrueDofs(Phi);
    mesh_cond.Transfer(phi_gf, phi_gf_cond);

    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream u_sock(vishost, visport);
        u_sock.precision(8);
        u_sock << "solution\n" << mesh_cond << u_gf << "window_title 'Deformation'" << endl;
        socketstream phi_sock(vishost, visport);
        phi_sock.precision(8);
        phi_sock << "solution\n" << mesh_cond << phi_gf_cond << "window_title 'Gravity Potential'" << endl;
    }

    delete mesh; 
    delete b1;
    delete b2;
    delete a11; 
    delete a12;
    delete a21;
    delete a22;

    return 0;
}

real_t rho_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    if (r > Constants::R){
        return 0.0;
    } else{
        real_t r_norm = r / Constants::R;
        //real_t theta = acos(coord[2] / r); // polar angle
        //real_t phi = atan2(coord[1], coord[0]); // azimuthal angle
        real_t rho_surface = 2.6e3; 
        real_t rho_center = 1.3e4;   
        real_t base_rho =  rho_center + (rho_surface - rho_center) * r_norm;
        return base_rho;
    }
}


real_t mu_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / 6371e3;
    real_t theta = acos(coord[2] / r); // polar angle
    real_t phi = atan2(coord[1], coord[0]); // azimuthal angle
    real_t mu_surface = 70e9;  // Pa
    real_t mu_center = 140e9;    // Pa
    real_t base_mu =  mu_center + (mu_surface - mu_center) * r_norm;
    real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 0.05 * sin(2.0 * phi);
    return base_mu * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t lamb_func(const Vector &coord)
{
    real_t r = coord.Norml2();
    real_t r_norm = r / 6371e3;
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    real_t lamb_surface = 100e9;
    real_t lamb_center = 300e9;   
    real_t base_lamb = lamb_center + (lamb_surface - lamb_center) * r_norm;
    real_t polar_perturb = 0.015 * (1.0 + cos(2.0 * theta));
    real_t azimuthal_perturb = 0.05 * sin(2.0 * phi);
    return base_lamb * (1.0 + polar_perturb) * (1.0 + azimuthal_perturb);
}

real_t loading_func(const Vector &coord)
{
    real_t factor = 1e-1;
    real_t r = coord.Norml2();
    real_t theta = acos(coord[2] / r);
    real_t phi = atan2(coord[1], coord[0]);
    // Max loading at poles (glaciers): e.g., 10 MPa (~1 km ice)
    const real_t polar_load = 10e6;
    // Equatorial loading (oceans): e.g., 1 MPa (~100 m water depth)
    const real_t equator_load = 1e6;
    real_t base_load = (equator_load + polar_load) / 2.0 + (polar_load - equator_load) / 2.0 * cos(2.0 * theta);
    real_t azimuthal_perturb = 0.2 * sin(2.0 * phi);
    return -base_load * (1.0 + azimuthal_perturb) * factor;
}
