#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "giafem.hpp"

using namespace mfem;
using namespace std;
using namespace giafem;


class StrainInterpolator2 : public DiscreteInterpolator {
 public:
  StrainInterpolator2() {}

  void AssembleElementMatrix2(const FiniteElement& displacement_fe,
                              const FiniteElement& strain_fe,
                              ElementTransformation& Trans,
                              DenseMatrix& elmat) override {
    auto dim = displacement_fe.GetDim();
    auto displacement_dof = displacement_fe.GetDof();
    auto strain_dof = strain_fe.GetDof();
    auto dshape = DenseMatrix(displacement_dof, dim);
    elmat.SetSize(dim * (dim + 1) * strain_dof / 2, dim * displacement_dof);
    elmat = 0.;

    const auto& nodes = strain_fe.GetNodes();
    for (auto i = 0; i < strain_dof; i++) {
      const IntegrationPoint& ip = nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      displacement_fe.CalcPhysDShape(Trans, dshape);

      constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);

      if (dim == 2) {
        for (auto j = 0; j < displacement_dof; j++) {
          // e_{00} = u_{0,0}
          elmat(i, j) = dshape(j, 0);

          // e_{10} = 0.5 * (u_{0,1} + u_{1,0})
          elmat(i + strain_dof, j) = half * dshape(j, 1);
          elmat(i + strain_dof, j + displacement_dof) = half * dshape(j, 0);

          // e_{11} = u_{1,1}
          elmat(i + 2 * strain_dof, j + displacement_dof) = dshape(j, 1);
        }
      } else {
        for (auto j = 0; j < displacement_dof; j++) {
          // e_{00} = u_{0,0}
          elmat(i, j) = dshape(j, 0);

          // e_{10} = 0.5 * (u_{0,1} + u_{1,0})
          elmat(i + strain_dof, j) = half * dshape(j, 1);
          elmat(i + strain_dof, j + displacement_dof) = half * dshape(j, 0);

          // e_{20} = 0.5 * (u_{0,2} + u_{2,0})
          elmat(i + 2 * strain_dof, j) = 0.5 * dshape(j, 2);
          elmat(i + 2 * strain_dof, j + 2 * displacement_dof) =
              0.5 * dshape(j, 0);

          // e_{11} = u_{1,1}
          elmat(i + 3 * strain_dof, j + displacement_dof) = dshape(j, 1);

          // e_{21} = 0.5 *( (u_{1,2} + u_{2,1})
          elmat(i + 4 * strain_dof, j + displacement_dof) = 0.5 * dshape(j, 2);
          elmat(i + 4 * strain_dof, j + 2 * displacement_dof) =
              0.5 * dshape(j, 1);

          // e_{22} = u_{2,2}
          elmat(i + 5 * strain_dof, j + 2 * displacement_dof) = dshape(j, 2);
        }
      }
    }
  }
};

class DeviatoricStrainInterpolator : public DiscreteInterpolator {
 public:
  DeviatoricStrainInterpolator() {}

  void AssembleElementMatrix2(const FiniteElement& displacement_fe,
                              const FiniteElement& deviatoric_strain_fe,
                              ElementTransformation& Trans,
                              DenseMatrix& elmat) override {
    auto dim = displacement_fe.GetDim();
    auto displacement_dof = displacement_fe.GetDof();
    auto deviatoric_strain_dof = deviatoric_strain_fe.GetDof();

    auto dshape = DenseMatrix(displacement_dof, dim);
    elmat.SetSize(dim * (dim + 1) * deviatoric_strain_dof / 2 - 1,
                  dim * displacement_dof);
    elmat = 0.;

    const auto& nodes = deviatoric_strain_fe.GetNodes();
    for (auto i = 0; i < deviatoric_strain_dof; i++) {
      const IntegrationPoint& ip = nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      displacement_fe.CalcPhysDShape(Trans, dshape);

      constexpr auto half = static_cast<real_t>(1) / static_cast<real_t>(2);
      constexpr auto third = static_cast<real_t>(1) / static_cast<real_t>(3);
      constexpr auto twoThirds =
          static_cast<real_t>(2) / static_cast<real_t>(3);
      if (dim == 2) {
        for (auto j = 0; j < displacement_dof; j++) {
          // d_{00} =   0.5 *(u_{0,0} - u_{1,1})
          elmat(i, j) = half * dshape(j, 0);
          elmat(i, j + displacement_dof) = -half * dshape(j, 1);

          // d_{10} = 0.5 *(u_{1,0} + u_{0,1})
          elmat(i + deviatoric_strain_dof, j) = half * dshape(j, 1);
          elmat(i + deviatoric_strain_dof, j + displacement_dof) =
              half * dshape(j, 0);
        }
      } else {
        for (auto j = 0; j < displacement_dof; j++) {
          // d_{00} =   (2/3) * u_{0,0} - (1/3) * (u_{1,1} + u_{2,2})
          elmat(i, j) = twoThirds * dshape(j, 0);
          elmat(i, j + displacement_dof) = -third * dshape(j, 1);
          elmat(i, j + 2 * displacement_dof) = -third * dshape(j, 2);

          // d_{10} = 0.5 *(u_{1,0} + u_{0,1})
          elmat(i + deviatoric_strain_dof, j) = half * dshape(j, 1);
          elmat(i + deviatoric_strain_dof, j + displacement_dof) =
              half * dshape(j, 0);

          // d_{20} = 0.5 *(u_{2,0} + u_{0,2})
          elmat(i + 2 * deviatoric_strain_dof, j) = half * dshape(j, 2);
          elmat(i + 2 * deviatoric_strain_dof, j + 2 * displacement_dof) =
              half * dshape(j, 0);

          // d_{11} = (2/3) * u_{1,1} - (1/3) * (u_{0,0} + u_{2,2})
          elmat(i + 3 * deviatoric_strain_dof, j) = -third * dshape(j, 0);
          elmat(i + 3 * deviatoric_strain_dof, j + displacement_dof) =
              twoThirds * dshape(j, 1);
          elmat(i + 3 * deviatoric_strain_dof, j + 2 * displacement_dof) =
              -third * dshape(j, 2);

          // d_{21} = 0.5 *(u_{2,1} + u_{1,2})
          elmat(i + 4 * deviatoric_strain_dof, j + displacement_dof) =
              half * dshape(j, 2);
          elmat(i + 4 * deviatoric_strain_dof, j + 2 * displacement_dof) =
              half * dshape(j, 1);
        }
      }
    }
  }
};

int main(int argc, char* argv[]) {
  const char* mesh_file = "../data/star.mesh";
  int order = 2;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  {
    int ref_levels = (int)floor(log(10000. / mesh.GetNE()) / log(2.) / dim);
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  auto H1 = H1_FECollection(order, dim);
  auto L2 = L2_FECollection(order - 1, dim);

  auto scalar_fes = FiniteElementSpace(&mesh, &L2);

  auto displacement_fes = FiniteElementSpace(&mesh, &H1, dim);

  auto deformation_dim = dim * dim;
  auto deformation_fes = FiniteElementSpace(&mesh, &L2, deformation_dim);

  auto strain_dim = dim * (dim + 1) / 2;
  auto strain_fes = FiniteElementSpace(&mesh, &L2, strain_dim);

  auto deviatoric_strain_dim = strain_dim - 1;
  auto deviatoric_strain_fes =
      FiniteElementSpace(&mesh, &L2, deviatoric_strain_dim);

  auto i = 0;
  auto j = 0;
  auto displacement_function = [dim, i, j](const Vector& x) {
    auto dim = x.Size();
    auto u = Vector(dim);
    u = 0.;
    u(0) = x(0);
    return u;
  };

  auto deformation_gradient_function = [dim, i, j](const Vector& x) {
    auto dim = x.Size();
    auto F = DenseMatrix(dim, dim);
    F = 0.;
    F(0, 0) = 1;
    return F;
  };

  auto uF = VectorFunctionCoefficient(
      dim, [&displacement_function](const Vector& x, Vector& u) {
        u = displacement_function(x);
      });

  auto FF = VectorFunctionCoefficient(
      deformation_dim,
      [&deformation_gradient_function](const Vector& x, Vector& F) {
        auto dim = x.Size();
        F.SetSize(dim * dim);
        auto Fmat = DenseMatrix(F.GetData(), dim, dim);
        Fmat = deformation_gradient_function(x);
      });

  auto EF = VectorFunctionCoefficient(
      strain_dim, [&deformation_gradient_function](const Vector& x, Vector& E) {
        auto dim = x.Size();
        E.SetSize(dim * (dim + 1) / 2);
        auto F = deformation_gradient_function(x);
        F.Symmetrize();
        E[0] = F(0, 0);
        E[1] = F(1, 0);
        if (dim == 2) {
          E[2] = F(1, 1);
        } else {
          E[2] = F(2, 0);
          E[3] = F(1, 1);
          E[4] = F(2, 1);
          E[5] = F(2, 2);
        }
      });

  auto DF = VectorFunctionCoefficient(
      deviatoric_strain_dim,
      [&deformation_gradient_function](const Vector& x, Vector& D) {
        auto dim = x.Size();
        D.SetSize(dim * (dim + 1) / 2 - 1);
        auto F = deformation_gradient_function(x);
        F.Symmetrize();
        auto trace = F.Trace();
        D[0] = F(0, 0) - trace / dim;
        D[1] = F(1, 0);
        if (dim == 3) {
          D[2] = F(2, 0);
          D[3] = F(1, 1) - trace / dim;
          D[4] = F(2, 1);
        }
      });

  auto u = GridFunction(&displacement_fes);
  u.ProjectCoefficient(uF);

  auto b = DiscreteLinearOperator(&displacement_fes, &deformation_fes);
  b.AddDomainIntegrator(new GradInterpolator(dim));
  b.Assemble();

  auto F = GridFunction(&deformation_fes);
  b.Mult(u, F);
  cout << "L2 error for deformation gradient = " << F.ComputeL2Error(FF)
       << endl;

  auto c = DiscreteLinearOperator(&displacement_fes, &strain_fes);
  c.AddDomainInterpolator(new StrainInterpolator(dim));
  c.Assemble();

  auto E = GridFunction(&strain_fes);
  c.Mult(u, E);
  cout << "L2 error for strain = " << E.ComputeL2Error(EF) << endl;

  auto d = DiscreteLinearOperator(&displacement_fes, &deviatoric_strain_fes);
  d.AddDomainInterpolator(new DevStrainInterpolator(dim));
  d.Assemble();

  auto D = GridFunction(&deviatoric_strain_fes);
  d.Mult(u, D);
  cout << "L2 error for deviatoric strain = " << D.ComputeL2Error(DF) << endl;

  ofstream deformation_ofs("deformation.gf");
  deformation_ofs.precision(8);
  F.Save(deformation_ofs);

  ofstream strain_ofs("strain.gf");
  strain_ofs.precision(8);
  E.Save(strain_ofs);

  ofstream deviatoric_strain_ofs("deviatoric_strain.gf");
  deviatoric_strain_ofs.precision(8);
  D.Save(deviatoric_strain_ofs);

  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);

  ofstream displacement_ofs("displacement.gf");
  displacement_ofs.precision(8);
  u.Save(displacement_ofs);
}
