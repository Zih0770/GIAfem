#include <cassert>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <numbers>

#include "mfem.hpp"
#include "giafem.hpp"

using namespace mfem;
using namespace giafem;
using namespace std;
const auto pi = numbers::pi_v<real_t>;

class MatrixGradIntegrator : public LinearFormIntegrator {
 private:
  MatrixCoefficient& _M;
  Vector v;
  DenseMatrix dshape, m;

 public:
  MatrixGradIntegrator(MatrixCoefficient& M,
                       const IntegrationRule* ir = nullptr)
      : LinearFormIntegrator(ir), _M{M} {}

  void AssembleRHSElementVect(const FiniteElement& el,
                              ElementTransformation& Tr,
                              Vector& elvect) override {
    auto dim = el.GetDim();
    auto vdim = _M.GetHeight();
    auto dof = el.GetDof();
    MFEM_ASSERT(_M.GetWidth() == dim,
                "Width of matrix coefficient must equal spatial dimension");

    dshape.SetSize(dof, dim);
    elvect.SetSize(dof * vdim);
    elvect = 0.0;
    v.SetSize(dof * vdim);
    auto vm = DenseMatrix(v.GetData(), dof, vdim);

    const IntegrationRule* ir = IntRule;
    if (ir == nullptr) {
      int intorder = 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
    }

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint& ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      auto factor = Tr.Weight() * ip.weight;
      el.CalcPhysDShape(Tr, dshape);
      _M.Eval(m, Tr, ip);
      MultABt(dshape, m, vm);
      elvect.Add(factor, v);
    }
  }

  using LinearFormIntegrator::AssembleRHSElementVect;
};

int main(int argc, char* argv[]) {
  // Parse command-line options.
  auto mesh_file = std::string("../data/star.mesh");
  int order = 2;

  auto args = mfem::OptionsParser(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return 1;
  }
  args.PrintOptions(std::cout);

  // Read the mesh from the given mesh file.
  auto mesh = mfem::Mesh(mesh_file, 1, 1);
  // auto mesh = mfem::Mesh::MakeCartesian2D(10, 10, Element::TRIANGLE);

  auto dim = mesh.Dimension();
  {
    int ref_levels = (int)floor(log(500. / mesh.GetNE()) / log(2.) / dim);
    for (int l = 0; l < ref_levels; l++) {
      mesh.UniformRefinement();
    }
  }

  // Define the vector finite element space.
  auto H1 = H1_FECollection(order + 1, dim);
  auto fes = FiniteElementSpace(&mesh, &H1, dim);

  // Set up a matrix coefficient function.
  auto m = MatrixFunctionCoefficient(dim, [](const Vector& x, DenseMatrix& m) {
    auto dim = x.Size();
    m.SetSize(dim);
    m = real_t{0};
    for (auto i = 0; i < dim; i++) {
      m(i, i) = 1;
    }
  });

  // Set up the linearform.
  auto b = LinearForm(&fes);
  b.AddDomainIntegrator(new MatrixGradLFIntegrator(m));
  b.Assemble();

  // Set up a vector field.
  auto f = VectorFunctionCoefficient(dim, [&](const Vector& x, Vector& y) {
    auto x0 = x(0);
    auto x1 = x(1);
    y.SetSize(x.Size());
    y(0) = x0;
    y(1) = x1;
  });
  auto x = GridFunction(&fes);
  x.ProjectCoefficient(f);

  cout << b(x) << endl;

  // Define scalar finite element space.
  auto L2 = L2_FECollection(order, dim);
  auto scalarFES = FiniteElementSpace(&mesh, &L2);

  // Set up the associated linear form.
  auto c = LinearForm(&scalarFES);
  auto one = ConstantCoefficient(1);
  c.AddDomainIntegrator(new DomainLFIntegrator(one));
  c.Assemble();

  // Set up scalar field as the divergence of the first.
  auto h = FunctionCoefficient([&](const Vector& x) { return 2; });
  auto y = GridFunction(&scalarFES);
  y.ProjectCoefficient(h);

  cout << c(y) << endl;

  std::ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh.Print(mesh_ofs);

  std::ofstream sol_ofs("div.gf");
  sol_ofs.precision(8);
  y.Save(sol_ofs);
}
