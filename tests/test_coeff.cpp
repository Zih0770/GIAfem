#include "mfem.hpp"
#include "giafem.hpp"
#include <iostream>
#include <cmath>

using namespace mfem;
using namespace giafem;
using namespace std;

// Analytic matrix coefficient for ∇u
class AnalyticGradU : public MatrixCoefficient
{
public:
    AnalyticGradU() : MatrixCoefficient(3,3) { }
    virtual void Eval(DenseMatrix &A,
                      ElementTransformation &T,
                      const IntegrationPoint &ip) override
    {
        // Compute physical point
        Vector x(3);
        T.Transform(ip, x);
        A.SetSize(3,3);
        A(0,0) = M_PI * cos(M_PI*x(0)) * exp(x(1));
        A(0,1) = sin(M_PI*x(0)) * exp(x(1));
        A(0,2) = 0.0;

        A(1,0) = 0.0;
        A(1,1) = -M_PI * sin(M_PI*x(1)) * x(2)*x(2);
        A(1,2) = 2.0 * x(2) * cos(M_PI*x(1));

        A(2,0) = x(1) * x(2);
        A(2,1) = x(0) * x(2);
        A(2,2) = x(0) * x(1);
    }
};

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

    // Build mesh and vector H1 space
    Mesh mesh = Mesh::MakeCartesian3D(16,16,16,
                                      Element::TETRAHEDRON, 1.0,1.0,1.0);
    int dim = mesh.Dimension();
    H1_FECollection fec(order, dim);
    FiniteElementSpace u_space(&mesh, &fec, dim);

    // Project analytic u
    VectorFunctionCoefficient u_exact(dim,
      [](const Vector &x, Vector &u)
      {
          u(0) = sin(M_PI*x(0)) * exp(x(1));
          u(1) = cos(M_PI*x(1)) * x(2)*x(2);
          u(2) = x(0)*x(1)*x(2);
      }
    );
    GridFunction u_gf(&u_space);
    u_gf.ProjectCoefficient(u_exact);

    // Custom coefficient from GridFunction
    GradientVectorGridFunctionCoefficient grad_u(&u_gf);
    // Analytic coefficient
    AnalyticGradU analytic;

    // Compute L2 error: ∫ ||grad_u - analytic||_F^2
    real_t err_sq = 0.0;
    real_t norm_sq = 0.0;
    int order_q = 2*fec.GetOrder() + 3;
    for (int e = 0; e < mesh.GetNE(); e++)
    {
        ElementTransformation *T = mesh.GetElementTransformation(e);
        const IntegrationRule &ir =
            IntRules.Get(mesh.GetElement(e)->GetGeometryType(), order_q);
        for (int i = 0; i < ir.GetNPoints(); i++)
        {
            const IntegrationPoint &ip = ir.IntPoint(i);
            T->SetIntPoint(&ip);
            real_t w = ip.weight * T->Weight();

            DenseMatrix M_h(dim, dim), M_a(dim, dim);
            grad_u.Eval(M_h, *T, ip);
            analytic.Eval(M_a, *T, ip);

            // Frobenius norm squared of difference
            for (int r = 0; r < dim; r++)
            {
                for (int c = 0; c < dim; c++)
                {
                    real_t diff = M_h(r,c) - M_a(r,c);
                    err_sq += w * diff * diff;
                    norm_sq += w * M_a(r,c) * M_a(r,c);
                }
            }
        }
    }
    real_t abs_err = sqrt(err_sq);
    real_t rel_err = (norm_sq > 0 ? sqrt(err_sq/norm_sq) : 0.0);
    cout << "Relative L2-error of gradient coefficient: " << rel_err << endl;

    return 0;
}

