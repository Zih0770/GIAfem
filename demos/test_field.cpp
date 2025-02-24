#include "mfem.hpp"
#include "giafem.hpp"
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;
using namespace giafem;

int main()
{
    // Define the computational domain: a simple 2D square mesh
    Mesh mesh = Mesh::MakeCartesian2D(16, 16, Element::QUADRILATERAL, true, 1.0, 1.0);
    int dim = mesh.Dimension();

    // Define finite element space
    int order = 2;
    H1_FECollection fec(order, dim);
    FiniteElementSpace fespace(&mesh, &fec, dim);

    // Define a GridFunction to hold the displacement field
    GridFunction u(&fespace);
    u = 0.0;

    // Define linear displacement field u_x = α * x, u_y = β * y
    double alpha = 1.0, beta = 0.5;
    VectorFunctionCoefficient u_func(dim, [&](const Vector &x, Vector &u_val)
    {
        u_val(0) = alpha * x(0); // u_x = α * x
        u_val(1) = beta * x(1);  // u_y = β * y
    });

    u.ProjectCoefficient(u_func); // Apply the function to the GridFunction

    cout << "Integration Point-wise Strain Comparison:\n";
    for (int i = 0; i < mesh.GetNE(); i++)
    {
        ElementTransformation *Tr = mesh.GetElementTransformation(i);
        const IntegrationRule &ir = IntRules.Get(mesh.GetElementBaseGeometry(i), 2 * order + 1);

        for (int d = 0; d < ir.GetNPoints(); d++) // Loop over integration points
        {
            const IntegrationPoint &ip = ir.IntPoint(d);
            Tr->SetIntPoint(&ip);

            DenseMatrix strain(dim, dim), dev_strain(dim, dim);
            FieldUtils::Strain_ip(u, *Tr, ip, strain);
            FieldUtils::DevStrain_ip(u, *Tr, ip, dev_strain);

            // Compute analytical strain at this integration point
            DenseMatrix strain_exact(dim, dim);
            strain_exact = 0.0;
            strain_exact(0, 0) = alpha; // ∂u_x / ∂x
            strain_exact(1, 1) = beta;  // ∂u_y / ∂y

            // Compute analytical deviatoric strain
            DenseMatrix dev_strain_exact(dim, dim);
            dev_strain_exact = strain_exact;
            double trace = (alpha + beta) / dim;
            dev_strain_exact(0, 0) -= trace;
            dev_strain_exact(1, 1) -= trace;

            // Print results for each integration point
            cout << "Element " << i << ", Integration Point " << d << ":\n";
            cout << "Numerical Strain:\n";
            strain.Print(cout);
            cout << "Exact Strain:\n";
            strain_exact.Print(cout);

            cout << "Numerical Deviatoric Strain:\n";
            dev_strain.Print(cout);
            cout << "Exact Deviatoric Strain:\n";
            dev_strain_exact.Print(cout);

            // Compute error for numerical validation
            double error = 0.0, dev_error = 0.0;
            for (int a = 0; a < dim; a++)
            {
                for (int b = 0; b < dim; b++)
                {
                    error += pow(strain(a, b) - strain_exact(a, b), 2);
                    dev_error += pow(dev_strain(a, b) - dev_strain_exact(a, b), 2);
                }
            }
            error = sqrt(error);
            dev_error = sqrt(dev_error);

            cout << "Strain Error: " << error << endl;
            cout << "Deviatoric Strain Error: " << dev_error << endl;
            cout << "---------------------------------------\n";
        }
    }

    return 0;
}

