#include "mfem.hpp"
#include "giafem.hpp"
#include <iostream>

using namespace mfem;
using namespace giafem;
using namespace std;

int main(int argc, char *argv[])
{
    int order = 2;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
            "Order (degree) of the finite elements.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    Mesh mesh = Mesh::MakeCartesian3D(16, 16, 16,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0);
    int dim = mesh.Dimension();

    H1_FECollection fec(order, dim);
    // φ ∈ H¹ scalar (trial), u ∈ [H¹]³ (test)
    FiniteElementSpace phi_space(&mesh, &fec);
    FiniteElementSpace u_space(&mesh, &fec, dim);

    // Spatial Q(x,y,z) = x + y + z
    FunctionCoefficient q_coeff(
        [](const Vector &x) { return x(0) + x(1) + x(2); }
    );

    // Exact trial φ(x) = x³ + 2y³ + 3z³
    FunctionCoefficient phi_exact(
        [](const Vector &x)
        { return pow(x(0),3) + 2*pow(x(1),3) + 3*pow(x(2),3); }
    );
    // ⇒ ∇φ = (3x², 6y², 9z²)

    // More complicated test u(x):
    // u = ( sin(pi x)*exp(y), cos(pi y)*z^2, x*y*z )
    VectorFunctionCoefficient u_exact(dim,
        [&](const Vector &x, Vector &u)
        {
        u(0) = sin(M_PI*x(0)) * exp(x(1));
        u(1) = cos(M_PI*x(1)) * x(2)*x(2);
        u(2) = x(0) * x(1) * x(2);
        }
        );

    VectorFunctionCoefficient v_exact(dim,
            [](const Vector &x, Vector &v)
            {
            v(0) = exp(x(0)) * cos(M_PI*x(2));
            v(1) = x(0)*x(1) + sin(x(2));
            v(2) = x(1) * exp(-x(0)*x(0));
            }
            );

    GridFunction phi_gf(&phi_space);
    phi_gf.ProjectCoefficient(phi_exact);
    GridFunction u_gf(&u_space), v_gf(&u_space);
    u_gf.ProjectCoefficient(u_exact);
    v_gf.ProjectCoefficient(v_exact);

    MixedBilinearForm mbf(&phi_space, &u_space);
    mbf.AddDomainIntegrator(new GradientIntegrator(q_coeff));
    mbf.Assemble();
    mbf.Finalize();

    Vector phi_td, u_td;
    phi_gf.GetTrueDofs(phi_td);
    u_gf.GetTrueDofs(u_td);

    Vector v_td(u_space.GetTrueVSize());
    mbf.Mult(phi_td, v_td);

    real_t discrete = u_td * v_td;


    double analytic = 0.0;
    int int_order = 2*order + 4;
    for (int el = 0; el < mesh.GetNE(); ++el)
    {
        ElementTransformation *T = mesh.GetElementTransformation(el);
        const IntegrationRule *ir = &IntRules.Get(
                mesh.GetElement(el)->GetGeometryType(), int_order);
        for (int i = 0; i < ir->GetNPoints(); ++i)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            T->SetIntPoint(&ip);
            double w = ip.weight * T->Weight();
            Vector x(dim);
            T->Transform(ip, x);
            double Qv = x(0) + x(1) + x(2);
            double g0 = 3 * x(0) * x(0);
            double g1 = 6 * x(1) * x(1);
            double g2 = 9 * x(2) * x(2);
            double u0 = sin(M_PI*x(0)) * exp(x(1));
            double u1 = cos(M_PI*x(1)) * x(2)*x(2);
            double u2 = x(0) * x(1) * x(2);
            double dot = Qv * (g0*u0 + g1*u1 + g2*u2);
            analytic += w * dot;
        }
    }

    cout << "GradientIntegrator: u' · ∇φ" << endl;
    cout << "Discrete u'^T A φ = " << discrete << " "
         << "Analytic = " << analytic  << " "
         << "Relative error = " << fabs((discrete - analytic) / analytic) << endl;

    // Advector: Qu·∇φ'
    MixedBilinearForm advec_mbf(&u_space, &phi_space);
    advec_mbf.AddDomainIntegrator(new AdvectionIntegrator(q_coeff));
    advec_mbf.Assemble();
    advec_mbf.Finalize();

    Vector phi_td2, u_td2;
    phi_gf.GetTrueDofs(phi_td2);
    u_gf.GetTrueDofs(u_td2);

    Vector w_td(phi_space.GetTrueVSize());
    advec_mbf.Mult(u_td2, w_td);

    real_t advec_discrete = phi_td2 * w_td;

    real_t advec_analytic = 0.0;
    for (int el = 0; el < mesh.GetNE(); ++el)
    {
        ElementTransformation *T = mesh.GetElementTransformation(el);
        const IntegrationRule *ir = &IntRules.Get(
                mesh.GetElement(el)->GetGeometryType(), int_order);
        for (int i = 0; i < ir->GetNPoints(); ++i)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            T->SetIntPoint(&ip);
            real_t w = ip.weight * T->Weight();
            Vector x(dim);
            T->Transform(ip, x);
            real_t Qv = x(0) + x(1) + x(2);
            real_t g0 = 3 * x(0) * x(0);
            real_t g1 = 6 * x(1) * x(1);
            real_t g2 = 9 * x(2) * x(2);
            real_t u0 = sin(M_PI*x(0)) * exp(x(1));
            real_t u1 = cos(M_PI*x(1)) * x(2)*x(2);
            real_t u2 = x(0) * x(1) * x(2);
            real_t dot = Qv * (u0*g0 + u1*g1 + u2*g2);
            advec_analytic += w * dot;
        }
    }
    cout << "AdvectionIntegrator: u · ∇φ'" << endl;
    cout << "Discrete = " << advec_discrete << " "
        << "Analytic = " << advec_analytic << " "
        << "Relative error = " << fabs((advec_discrete - advec_analytic) / advec_analytic) << endl;

    //ProjectionGradientIntegrator
    // Q_vec(x, y, z) = (x², y², z²)
    VectorFunctionCoefficient Q_vec(dim,
            [](const Vector &x, Vector &qv) {
            qv(0) = x(0)*x(0);
            qv(1) = x(1)*x(1);
            qv(2) = x(2)*x(2);
            }
            );

    // ∇Q_vec = diag(2x, 2y, 2z)
    MatrixFunctionCoefficient dQ_vec(dim,
            [](const Vector &x, DenseMatrix &dqv) {
            dqv.SetSize(3);
            dqv = 0.0;
            dqv(0,0) = 2*x(0);
            dqv(1,1) = 2*x(1);
            dqv(2,2) = 2*x(2);
            }
            );

    BilinearForm pg_form(&u_space);
    pg_form.AddDomainIntegrator(new ProjectionGradientIntegrator(q_coeff, Q_vec, dQ_vec));
    pg_form.Assemble();
    pg_form.Finalize();

    Vector u_td_3, v_td_3;
    u_gf.GetTrueDofs(u_td_3);
    v_gf.GetTrueDofs(v_td_3);

    Vector PG_td(u_space.GetTrueVSize());
    pg_form.Mult(u_td_3, PG_td);

    real_t pg_discrete = v_td_3 * PG_td;

    real_t pg_analytic = 0.0;
    for (int el = 0; el < mesh.GetNE(); ++el)
    {
        ElementTransformation *T = mesh.GetElementTransformation(el);
        const IntegrationRule *ir = &IntRules.Get(
                mesh.GetElement(el)->GetGeometryType(), int_order);

        for (int i = 0; i < ir->GetNPoints(); ++i)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            T->SetIntPoint(&ip);
            real_t w = ip.weight * T->Weight();
            Vector x(dim); T->Transform(ip, x);

            real_t Qs = x(0) + x(1) + x(2);

            Vector u(dim), v(dim), Qv(dim), grad_uQv(dim);
            // u
            u(0) = sin(M_PI*x(0)) * exp(x(1));
            u(1) = cos(M_PI*x(1)) * x(2)*x(2);
            u(2) = x(0) * x(1) * x(2);

            // v
            v(0) = exp(x(0)) * cos(M_PI*x(2));
            v(1) = x(0)*x(1) + sin(x(2));
            v(2) = x(1) * exp(-x(0)*x(0));

            // Qv
            Qv(0) = x(0)*x(0);
            Qv(1) = x(1)*x(1);
            Qv(2) = x(2)*x(2);

            // ∇(u·Qv)
            grad_uQv(0) = cos(M_PI*x(0))*M_PI*exp(x(1)) * x(0)*x(0) + exp(x(1))*sin(M_PI*x(0)) * 2*x(0) + x(1)*pow(x(2), 3);
            grad_uQv(1) = -M_PI*sin(M_PI*x(1))*x(2)*x(2)*x(1)*x(1) + cos(M_PI*x(1))*x(2)*x(2) * 2*x(1) + x(0)*pow(x(2), 3) + sin(M_PI*x(0))*pow(x(0), 2)*exp(x(1));
            grad_uQv(2) = 3*x(0)*x(1)*pow(x(2),2) + 2*cos(M_PI*x(1))*pow(x(1),2)*x(2);

            real_t integrand = Qs * (grad_uQv * v);
            pg_analytic += w * integrand;
        }
    }

    cout << "ProjectionGradientIntegrator: v · ∇(u · Q_vec)" << endl;
    cout << "Discrete vᵀ A_pg u = " << pg_discrete << " "
        << "Analytic = " << pg_analytic << " "
        << "Relative error = " << fabs((pg_discrete - pg_analytic) / pg_analytic) << endl;

    //AdvectionProjectionIntegrator
    BilinearForm ap_form(&u_space);
    ap_form.AddDomainIntegrator(new AdvectionProjectionIntegrator(q_coeff, Q_vec, dQ_vec));
    ap_form.Assemble();
    ap_form.Finalize();

    Vector u_td_4, v_td_4;
    u_gf.GetTrueDofs(u_td_4);
    v_gf.GetTrueDofs(v_td_4);

    Vector AP_td(u_space.GetTrueVSize());
    ap_form.Mult(u_td_4, AP_td);

    real_t ap_discrete = v_td_4 * AP_td;

    real_t ap_analytic = 0.0;
    for (int el = 0; el < mesh.GetNE(); ++el)
    {
        ElementTransformation *T = mesh.GetElementTransformation(el);
        const IntegrationRule *ir = &IntRules.Get(
                mesh.GetElement(el)->GetGeometryType(), int_order);

        for (int i = 0; i < ir->GetNPoints(); ++i)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            T->SetIntPoint(&ip);
            real_t w = ip.weight * T->Weight();
            Vector x(dim); T->Transform(ip, x);

            real_t Qs = x(0) + x(1) + x(2);

            Vector u(dim), v(dim), Qv(dim), grad_vQv(dim);
            u(0) = sin(M_PI*x(0)) * exp(x(1));
            u(1) = cos(M_PI*x(1)) * x(2)*x(2);
            u(2) = x(0) * x(1) * x(2);

            v(0) = exp(x(0)) * cos(M_PI*x(2));
            v(1) = x(0)*x(1) + sin(x(2));
            v(2) = x(1) * exp(-x(0)*x(0));

            Qv(0) = x(0)*x(0);
            Qv(1) = x(1)*x(1);
            Qv(2) = x(2)*x(2);

            // grad(v · Qv)
            grad_vQv(0) = cos(M_PI*x(2)) * (exp(x(0)) * pow(x(0), 2) + 2 * x(0) * exp(x(0))) + pow(x(1), 3) - 2 * x(0) * x(1) * pow(x(2), 2) * exp(-pow(x(0), 2));

            grad_vQv(1) = 3 * x(0) * pow(x(1), 2) + 2 * x(1) * sin(x(2)) + pow(x(2), 2) * exp(-pow(x(0), 2));

            grad_vQv(2) = -exp(x(0)) * sin(M_PI*x(2)) * M_PI * x(0)*x(0) + pow(x(1), 2) * cos(x(2)) + 2 * x(1) * x(2) * exp(-pow(x(0), 2));

            real_t integrand = Qs * (grad_vQv * u);
            ap_analytic += w * integrand;
        }
    }

    cout << "ProjectionGradientIntegrator: u · ∇(v · Q_vec)" << endl;
    cout << "Discrete vᵀ A_ap u = " << ap_discrete << " "
        << "Analytic = " << ap_analytic << " "
        << "Relative error = " << fabs((ap_discrete - ap_analytic) / ap_analytic) << endl;

    //DivergenceVectorIntegrator
    BilinearForm dv_form(&u_space);
    dv_form.AddDomainIntegrator(new DivergenceVectorIntegrator(q_coeff, Q_vec));
    dv_form.Assemble();
    dv_form.Finalize();

    Vector u_td_5, v_td_5;
    u_gf.GetTrueDofs(u_td_5);
    v_gf.GetTrueDofs(v_td_5);

    Vector DV_td(u_space.GetTrueVSize());
    dv_form.Mult(u_td_5, DV_td);

    real_t dv_discrete = v_td_5 * DV_td;

    real_t dv_analytic = 0.0;
    for (int el = 0; el < mesh.GetNE(); ++el)
    {
        ElementTransformation *T = mesh.GetElementTransformation(el);
        const IntegrationRule *ir = &IntRules.Get(
                mesh.GetElement(el)->GetGeometryType(), int_order);

        for (int i = 0; i < ir->GetNPoints(); ++i)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            T->SetIntPoint(&ip);
            real_t w = ip.weight * T->Weight();
            Vector x(dim); T->Transform(ip, x);

            real_t Qs = x(0) + x(1) + x(2);

            Vector u(dim), v(dim), Qv(dim), grad_vQv(dim);
            u(0) = sin(M_PI*x(0)) * exp(x(1));
            u(1) = cos(M_PI*x(1)) * x(2)*x(2);
            u(2) = x(0) * x(1) * x(2);

            v(0) = exp(x(0)) * cos(M_PI*x(2));
            v(1) = x(0)*x(1) + sin(x(2));
            v(2) = x(1) * exp(-x(0)*x(0));

            Qv(0) = x(0)*x(0);
            Qv(1) = x(1)*x(1);
            Qv(2) = x(2)*x(2);

            // div(u)
            real_t div_u = M_PI * cos(M_PI*x(0)) * exp(x(1)) - M_PI * sin(M_PI*x(1)) * pow(x(2), 2) + x(0)*x(1);

            // Q_vec · v
            real_t vec_v = cos(M_PI*x(2)) * exp(x(0)) * pow(x(0), 2) + pow(x(1), 2) * (x(0)*x(1) + sin(x(2))) + x(1) * pow(x(2), 2) * exp(-pow(x(0), 2));

            real_t integrand = Qs * div_u * vec_v;
            dv_analytic += w * integrand;
        }
    }

    cout << "DivergenceVectorIntegrator: ∇ · u Q_vec · v" << endl;
    cout << "Discrete vᵀ A_dv u = " << dv_discrete << " "
        << "Analytic = " << dv_analytic << " "
        << "Relative error = " << fabs((dv_discrete - dv_analytic) / dv_analytic) << endl;
    
    //ProjectionDivergenceIntegrator
    BilinearForm pd_form(&u_space);
    pd_form.AddDomainIntegrator(new ProjectionDivergenceIntegrator(q_coeff, Q_vec));
    pd_form.Assemble();
    pd_form.Finalize();

    Vector u_td_6, v_td_6;
    u_gf.GetTrueDofs(u_td_6);
    v_gf.GetTrueDofs(v_td_6);

    Vector PD_td(u_space.GetTrueVSize());
    pd_form.Mult(u_td_6, PD_td);

    real_t pd_discrete = v_td_6 * PD_td;

    real_t pd_analytic = 0.0;
    for (int el = 0; el < mesh.GetNE(); ++el)
    {
        ElementTransformation *T = mesh.GetElementTransformation(el);
        const IntegrationRule *ir = &IntRules.Get(
                mesh.GetElement(el)->GetGeometryType(), int_order);

        for (int i = 0; i < ir->GetNPoints(); ++i)
        {
            const IntegrationPoint &ip = ir->IntPoint(i);
            T->SetIntPoint(&ip);
            real_t w = ip.weight * T->Weight();
            Vector x(dim); T->Transform(ip, x);

            real_t Qs = x(0) + x(1) + x(2);

            Vector u(dim), v(dim), Qv(dim), grad_vQv(dim);
            u(0) = sin(M_PI*x(0)) * exp(x(1));
            u(1) = cos(M_PI*x(1)) * x(2)*x(2);
            u(2) = x(0) * x(1) * x(2);

            v(0) = exp(x(0)) * cos(M_PI*x(2));
            v(1) = x(0)*x(1) + sin(x(2));
            v(2) = x(1) * exp(-x(0)*x(0));

            Qv(0) = x(0)*x(0);
            Qv(1) = x(1)*x(1);
            Qv(2) = x(2)*x(2);

            // div(v)
            real_t div_v = cos(M_PI*x(2)) * exp(x(0)) + x(0);

            // Q_vec · u
            real_t vec_u = sin(M_PI*x(0)) * exp(x(1)) * pow(x(0), 2) + cos(M_PI*x(1)) * x(1)*x(1) * x(2)*x(2) + x(0) * x(1) * pow(x(2), 3);

            real_t integrand = Qs * vec_u * div_v;
            pd_analytic += w * integrand;
        }
    }

    cout << "ProjectionDivergenceIntegrator: ∇ · v Q_vec · u" << endl;
    cout << "Discrete vᵀ A_pd u = " << pd_discrete << " "
        << "Analytic = " << pd_analytic << " "
        << "Relative error = " << fabs((pd_discrete - pd_analytic) / pd_analytic) << endl;







    return 0;
}

