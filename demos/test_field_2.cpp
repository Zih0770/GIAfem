#include "mfem.hpp"
#include "giafem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;
using namespace giafem;

void AnalyticalDisplacement(const Vector &x, Vector &u)
{
    u.SetSize(x.Size());
    if (x.Size() == 2)
    {
        // 2D test: u = (x + y, x*y)
        u(0) = x(0) + x(1);
        u(1) = x(0) * x(1);
    }
    else if (x.Size() == 3)
    {
        // 3D test: u = (x + y + z, x*y + z, x*y*z)
        u(0) = x(0) + x(1) + x(2);
        u(1) = x(0) * x(1) + x(2);
        u(2) = x(0) * x(1) * x(2);
    }
}

void AnalyticalStrain(const Vector &x, Vector &strain)
{
    strain.SetSize(x.Size() == 2 ? 3 : 6);
    if (x.Size() == 2)
    {
        // 2D case
        strain(0) = 1.0;             
        strain(1) = 0.5 * (1 + x(1));
        strain(2) = x(0);
    }
    else if (x.Size() == 3)
    {
        // 3D case
        strain(0) = 1.0;               
        strain(1) = 0.5 * (1 + x(1));  
        strain(2) = 0.5 * (1 + x(1) * x(2));
        strain(3) = x(0);
        strain(4) = 0.5 * (1 + x(0) * x(2));
        strain(5) = x(0) * x(1);
    }
}

void AnalyticalDevStrain(const Vector &x, Vector &dev_strain)
{
    Vector strain(6);
    AnalyticalStrain(x, strain);
    double trace = (strain(0) + strain(3) + strain(5)) / 3.0;
    
    dev_strain.SetSize(5); // Deviatoric strain has 5 components in 3D
    dev_strain(0) = strain(0) - trace;
    dev_strain(1) = strain(1);
    dev_strain(2) = strain(2);
    dev_strain(3) = strain(3) - trace;
    dev_strain(4) = strain(4);
}

int main(int argc, char *argv[])
{
    Mesh mesh = Mesh::MakeCartesian3D(16, 16, 16, Element::HEXAHEDRON);
    int dim = mesh.Dimension();
    int vdim = dim;

    int order = 1;

    H1_FECollection fec(order, dim);
    L2_FECollection fec_L2(order, dim);
    FiniteElementSpace fes(&mesh, &fec, vdim);
    FiniteElementSpace fes_strain(&mesh, &fec_L2, vdim * (vdim + 1) / 2);
    FiniteElementSpace fes_dev_strain(&mesh, &fec_L2, vdim * (vdim + 1) / 2 - 1);

    GridFunction u(&fes);
    GridFunction strain(&fes_strain);
    GridFunction dev_strain(&fes_dev_strain);

    VectorFunctionCoefficient u_coef(dim, AnalyticalDisplacement);
    u.ProjectCoefficient(u_coef);

    VectorFunctionCoefficient strain_coef(vdim * (vdim + 1) / 2, AnalyticalStrain);

    VectorFunctionCoefficient dev_strain_coef(vdim * (vdim + 1) / 2 - 1, AnalyticalDevStrain);

    // Compute strain and deviatoric strain
    FieldUtils::Strain(u, strain);
    FieldUtils::DevStrain(u, dev_strain);

    double strain_error = strain.ComputeL2Error(strain_coef);
    double dev_strain_error = dev_strain.ComputeL2Error(dev_strain_coef);

    cout << "Global strain error (L2 norm): " << strain_error << endl;
    cout << "Global deviatoric strain error (L2 norm): " << dev_strain_error << endl;

    return 0;
}

