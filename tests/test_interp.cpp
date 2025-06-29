#include "mfem.hpp"
#include "giafem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;
using namespace giafem;

static const double PI = M_PI;

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

    VectorFunctionCoefficient u_coef(3,
            [](const Vector &x, Vector &u)
            {
            double sx = sin(PI*x[0]), cx = cos(PI*x[0]);
            double sy = sin(PI*x[1]), cy = cos(PI*x[1]);
            double sz = sin(PI*x[2]), cz = cos(PI*x[2]);
            u[0] = sx*cy*cz;
            u[1] = cx*sy*cz;
            u[2] = cx*cy*sz;
            });

    VectorFunctionCoefficient grad_coef(9,
            [](const Vector &x, Vector &g)
            {
            double sx = sin(PI*x[0]), cx = cos(PI*x[0]);
            double sy = sin(PI*x[1]), cy = cos(PI*x[1]);
            double sz = sin(PI*x[2]), cz = cos(PI*x[2]);
            // compute all ∂u_i/∂j
            double d00 =  PI*cx*cy*cz; // ∂u0/∂x
            double d01 = -PI*sx*sy*cz; // ∂u0/∂y
            double d02 = -PI*sx*cy*sz; // ∂u0/∂z
            double d10 = -PI*sx*sy*cz; // ∂u1/∂x
            double d11 =  PI*cx*cy*cz; // ∂u1/∂y
            double d12 = -PI*cx*sy*sz; // ∂u1/∂z
            double d20 = -PI*sx*cy*sz; // ∂u2/∂x
            double d21 = -PI*cx*sy*sz; // ∂u2/∂y
            double d22 =  PI*cx*cy*cz; // ∂u2/∂z

            // column-major flattening
            g[0] = d00;  g[1] = d10;  g[2] = d20;
            g[3] = d01;  g[4] = d11;  g[5] = d21;
            g[6] = d02;  g[7] = d12;  g[8] = d22;
            });

    VectorFunctionCoefficient grad_coef_reduced(6,
            [](const Vector &x, Vector &g)
            {
            double sx = sin(PI*x[0]), cx = cos(PI*x[0]);
            double sy = sin(PI*x[1]), cy = cos(PI*x[1]);
            double sz = sin(PI*x[2]), cz = cos(PI*x[2]);
            // compute all ∂u_i/∂j
            double d00 =  PI*cx*cy*cz; // ∂u0/∂x
            double d01 = -PI*sx*sy*cz; // ∂u0/∂y
            double d02 = -PI*sx*cy*sz; // ∂u0/∂z
            double d10 = -PI*sx*sy*cz; // ∂u1/∂x
            double d11 =  PI*cx*cy*cz; // ∂u1/∂y
            double d12 = -PI*cx*sy*sz; // ∂u1/∂z
            double d20 = -PI*sx*cy*sz; // ∂u2/∂x
            double d21 = -PI*cx*sy*sz; // ∂u2/∂y
            double d22 =  PI*cx*cy*cz; // ∂u2/∂z

            // column-major flattening
            g[0] = d00;  g[1] = d10;  g[2] = d20;
            g[3] = d11;  g[4] = d21;  g[5] = d22;
            });

    VectorFunctionCoefficient strain_coef_reduced(6,
            [](const Vector &x, Vector &e)
            {
            double sx = sin(PI*x[0]), cx = cos(PI*x[0]);
            double sy = sin(PI*x[1]), cy = cos(PI*x[1]);
            double sz = sin(PI*x[2]), cz = cos(PI*x[2]);
            double d00 =  PI*cx*cy*cz;
            double d01 = -PI*sx*sy*cz;
            double d02 = -PI*sx*cy*sz;
            double d11 =  PI*cx*cy*cz;
            double d12 = -PI*cx*sy*sz;
            double d22 =  PI*cx*cy*cz;
            // symmetric parts
            e[0] =  d00;              // (0,0)
            e[1] = 0.5*(d01 + d01);   // (1,0)
            e[2] = 0.5*(d02 + d02);   // (2,0)
            e[3] =  d11;              // (1,1)
            e[4] = 0.5*(d12 + d12);   // (2,1)
            e[5] =  d22;              // (2,2)
            });

    VectorFunctionCoefficient strain_coef(9,
            [](const Vector &x, Vector &e)
            {
            double sx = sin(PI*x[0]), cx = cos(PI*x[0]);
            double sy = sin(PI*x[1]), cy = cos(PI*x[1]);
            double sz = sin(PI*x[2]), cz = cos(PI*x[2]);
            double d00 =  PI*cx*cy*cz;
            double d01 = -PI*sx*sy*cz;
            double d02 = -PI*sx*cy*sz;
            double d11 =  PI*cx*cy*cz;
            double d12 = -PI*cx*sy*sz;
            double d22 =  PI*cx*cy*cz;
            // symmetric parts
            e[0] =  d00;              // (0,0)
            e[1] = 0.5*(d01 + d01);   // (1,0)
            e[2] = 0.5*(d02 + d02);   // (2,0)
            e[3] = e[1];
            e[4] =  d11;              // (1,1)
            e[5] = 0.5*(d12 + d12);   // (2,1)
            e[6] = e[2];
            e[7] = e[5];
            e[8] =  d22;              // (2,2)
            });


    VectorFunctionCoefficient devs_coef_reduced(5,
            [](const Vector &x, Vector &d)
            {
            double sx = sin(PI*x[0]), cx = cos(PI*x[0]);
            double sy = sin(PI*x[1]), cy = cos(PI*x[1]);
            double sz = sin(PI*x[2]), cz = cos(PI*x[2]);
            double d00 =  PI*cx*cy*cz;
            double d01 = -PI*sx*sy*cz;
            double d02 = -PI*sx*cy*sz;
            double d11 =  PI*cx*cy*cz;
            double d12 = -PI*cx*sy*sz;
            double d22 =  PI*cx*cy*cz;
            // symmetrize
            double s00 = d00;
            double s10 = 0.5*(d01 + d01);
            double s20 = 0.5*(d02 + d02);
            double s11 = d11;
            double s21 = 0.5*(d12 + d12);
            double s22 = d22;
            double tr  = s00 + s11 + s22;
            // build dev‐strain
            d[0] = s00 - tr/3.0;  // (0,0)
            d[1] = s10;           // (1,0)
            d[2] = s20;           // (2,0)
            d[3] = s11 - tr/3.0;  // (1,1)
            d[4] = s21;           // (2,1)
            });

        VectorFunctionCoefficient devs_coef(9,
            [](const Vector &x, Vector &d)
            {
            double sx = sin(PI*x[0]), cx = cos(PI*x[0]);
            double sy = sin(PI*x[1]), cy = cos(PI*x[1]);
            double sz = sin(PI*x[2]), cz = cos(PI*x[2]);
            double d00 =  PI*cx*cy*cz;
            double d01 = -PI*sx*sy*cz;
            double d02 = -PI*sx*cy*sz;
            double d11 =  PI*cx*cy*cz;
            double d12 = -PI*cx*sy*sz;
            double d22 =  PI*cx*cy*cz;
            // symmetrize
            double s00 = d00;
            double s10 = 0.5*(d01 + d01);
            double s20 = 0.5*(d02 + d02);
            double s11 = d11;
            double s21 = 0.5*(d12 + d12);
            double s22 = d22;
            double tr  = s00 + s11 + s22;
            // build dev‐strain
            d[0] = s00 - tr/3.0;  // (0,0)
            d[1] = s10;           // (1,0)
            d[2] = s20;           // (2,0)
            d[3] = d[1];
            d[4] = s11 - tr/3.0;  // (1,1)
            d[5] = s21;
            d[6] = d[2];
            d[7] = d[5];
            d[8] = s22 - tr/3.0;           
            });


    //const char *mesh_file = "mesh/Earth_space.msh";
    //Mesh *mesh = new Mesh(mesh_file, 1, 1);
    auto mesh = Mesh::MakeCartesian3D(16,16,16, Element::TETRAHEDRON);

    H1_FECollection   u_fec(order, mesh.Dimension());
    FiniteElementSpace u_fes(&mesh, &u_fec, 3);

    L2_FECollection   grad_fec(order-1, mesh.Dimension());
    FiniteElementSpace grad_fes_reduced(&mesh, &grad_fec, 6);
    FiniteElementSpace grad_fes(&mesh, &grad_fec, 9);

    L2_FECollection   strain_fec(order-1, mesh.Dimension());
    FiniteElementSpace strain_fes_reduced(&mesh, &strain_fec, 6);
    FiniteElementSpace strain_fes(&mesh, &strain_fec, 9);

    L2_FECollection   devs_fec(order-1, mesh.Dimension());
    FiniteElementSpace devs_fes_reduced(&mesh, &devs_fec, 5);
    FiniteElementSpace devs_fes(&mesh, &devs_fec, 9);

    // --- project and apply interpolators ---
    GridFunction U(&u_fes);
    U.ProjectCoefficient(u_coef);

    GridFunction G(&grad_fes), E(&strain_fes), D(&devs_fes), G_(&grad_fes_reduced), E_(&strain_fes_reduced), D_(&devs_fes_reduced);

    DiscreteLinearOperator gradOp(&u_fes, &grad_fes);
    gradOp.AddDomainIntegrator(new GradInterpolator(3, VecMode::FULL));
    gradOp.Assemble();

    DiscreteLinearOperator gradOp_(&u_fes, &grad_fes_reduced);
    gradOp_.AddDomainIntegrator(new GradInterpolator(3, VecMode::REDUCED));
    gradOp_.Assemble();

    DiscreteLinearOperator strainOp(&u_fes, &strain_fes);
    strainOp.AddDomainIntegrator(new StrainInterpolator(3, VecMode::FULL));
    strainOp.Assemble();
    
    DiscreteLinearOperator strainOp_(&u_fes, &strain_fes_reduced);
    strainOp_.AddDomainIntegrator(new StrainInterpolator(3, VecMode::REDUCED));
    strainOp_.Assemble();
   
    DiscreteLinearOperator devsOp(&u_fes, &devs_fes);
    devsOp.AddDomainIntegrator(new DevStrainInterpolator(3, VecMode::FULL));
    devsOp.Assemble();
   
    DiscreteLinearOperator devsOp_(&u_fes, &devs_fes_reduced);
    devsOp_.AddDomainIntegrator(new DevStrainInterpolator(3, VecMode::REDUCED));
    devsOp_.Assemble();


    gradOp.Mult(U, G);
    strainOp.Mult(U, E);
    devsOp.Mult(U, D);

    gradOp_.Mult(U, G_);
    strainOp_.Mult(U, E_);
    devsOp_.Mult(U, D_);


    // --- compute L2 errors against the analytic coefficients ---
    double errG = G.ComputeL2Error(grad_coef) / G.Norml2();
    double errE = E.ComputeL2Error(strain_coef)/ E.Norml2();
    double errD = D.ComputeL2Error(devs_coef) / D.Norml2();
    double errG_ = G_.ComputeL2Error(grad_coef_reduced) / G_.Norml2();
    double errE_ = E_.ComputeL2Error(strain_coef_reduced)/ E_.Norml2();
    double errD_ = D_.ComputeL2Error(devs_coef_reduced) / D_.Norml2();


    cout << "L2 error ∇u:   " << errG << "\n"
        << "L2 error ε(u): " << errE << "\n"
        << "L2 error devε: " << errD << "\n"
        << "L2 error ∇u_rd:   " << errG_ << "\n"
        << "L2 error ε(u)_rd: " << errE_ << "\n"
        << "L2 error devε_rd: " << errD_ << "\n";

    //delete mesh;

    return 0;

}
