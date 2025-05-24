#include "mfem.hpp"
#include "giafem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>

using namespace std;
using namespace mfem;
using namespace giafem;

/*
  5-Component Vector Field Relaxation on 3D Mesh
  -------------------------------------------------------------
  Solves the system:
    dm(x,t)/dt = (1/tau) [ d(x,t) - m(x,t) ],
  where m(x,t) and d(x,t) are vector fields in R^5 defined on a 3D mesh.

  Forcing:
    d(x,t) = d0(x) e^{-alp*t},
    d0_j(x) = (j+1) sin(π x) sin(π y) sin(π z),  j = 0,...,4.

  Analytical solution at each degree of freedom:
    m_ana(x,t) = m0(x) e^{-t/tau} + [ d0(x)/(1 - alp*tau) ] [ e^{-alp*t} - e^{-t/tau} ].

  Initial condition:
    m(x,0) = m0(x),
    m0_j(x) given by a smooth 5-vector FunctionCoefficient.

  Outputs error data in error.dat for L2( m_num - m_ana ).
*/

class CountingRelaxOp : public TimeDependentOperator
{
private:
    real_t tau, alpha;
    Vector d0_true;
    mutable int mult_calls;
public:
    CountingRelaxOp(real_t tau_, real_t alpha_, const Vector &d0)
      : TimeDependentOperator(d0.Size()), tau(tau_), alpha(alpha_), d0_true(d0), mult_calls(0) { }

    void ResetCounter() const { mult_calls = 0; }
    int GetCounter() const { return mult_calls; }

    void Mult(const Vector &m, Vector &dm_dt) const override
    {
        mult_calls++;
        real_t t = this->GetTime();
        int n = m.Size();
        for (int i = 0; i < n; i++)
        {
            real_t dval = d0_true(i) * exp(-alpha * t);
            dm_dt(i) = (dval - m(i)) / tau;
        }
    }

    void ImplicitSolve(const real_t dt, const Vector &m_old, Vector &dm_dt) override
    {
        mult_calls++;
        real_t t = this->GetTime() + dt;
        int n = m_old.Size();

        for (int i = 0; i < n; i++)
        {
            real_t d_next = d0_true(i) * exp(-alpha * t);
            dm_dt(i) = (d_next - m_old(i)) / tau / (1.0 + dt / tau);
        }
    }
};

int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/examples/beam-tet.mesh";
    real_t tau = 5.0, alpha = 1.5, t_final = 30.0;
    int solver_type = 1; // Default Forward Euler

    OptionsParser args(argc, argv);
    args.AddOption(&solver_type, "-s", "--solver", "ODE solver type");
    args.Parse();
    if (!args.Good()) { args.PrintUsage(cout); return 1; }

    Mesh mesh(mesh_file, 1, 1);
    int order = 2;
    H1_FECollection fec(order, mesh.Dimension());
    FiniteElementSpace fes(&mesh, &fec, 5);
    int size = fes.GetTrueVSize();

    Vector m(size), m0(size), d0(size);
    GridFunction m0_gf(&fes), d0_gf(&fes);

    VectorFunctionCoefficient m0_coeff(5,
      [](const Vector &x, Vector &v)
      {
        v(0)=sin(M_PI*x[0])*sin(M_PI*x[1])*sin(M_PI*x[2]);
        v(1)=cos(M_PI*x[0])*sin(M_PI*x[1])*sin(M_PI*x[2]);
        v(2)=sin(M_PI*x[0])*cos(M_PI*x[1])*sin(M_PI*x[2]);
        v(3)=sin(M_PI*x[0])*sin(M_PI*x[1])*cos(M_PI*x[2]);
        v(4)=cos(M_PI*x[0])*cos(M_PI*x[1])*cos(M_PI*x[2]);
      });
    m0_gf.ProjectCoefficient(m0_coeff);
    m0_gf.GetTrueDofs(m0); m = m0;

    VectorFunctionCoefficient d0_coeff(5,
      [](const Vector &x, Vector &v)
      {
        double S = sin(M_PI*x[0])*sin(M_PI*x[1])*sin(M_PI*x[2]);
        for (int j = 0; j < 5; ++j) v(j)=(j+1)*S;
      });
    d0_gf.ProjectCoefficient(d0_coeff);
    d0_gf.GetTrueDofs(d0);

    CountingRelaxOp oper(tau, alpha, d0);

    std::unique_ptr<ODESolver> ode_solver;

    if (solver_type != 99)
    {
        ode_solver = ODESolver::Select(solver_type);
    }
    else
    {
        ode_solver = std::make_unique<BaileySolver_test>();
    }

    ode_solver->Init(oper);

    ofstream fout("solver_"+to_string(solver_type)+"_tau.dat");
fout << "#tau dt N_calls L2_error\n";

vector<double> tau_values = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}; // Example tau values

double target_error = 1e-5;

for (double tau : tau_values)
{
    cout<<tau<<endl;
    oper = CountingRelaxOp(tau, alpha, d0); // Reinitialize operator for each tau
    ode_solver->Init(oper);

    double dt_low = 1e-6, dt_high = 5.0, dt = 1.0;
    real_t actual_error;

    for (int iter = 0; iter < 30; ++iter)
    {
        oper.ResetCounter();
        m = m0; double t = 0.0;

        while (t < t_final - 1e-8)
        {
            real_t h = min(dt, t_final - t);
            ode_solver->Step(m, t, h);
        }

        Vector ana(size);
        double e1 = exp(-t / tau), e2 = exp(-alpha * t), denom = 1 - alpha * tau;
        for (int i = 0; i < size; ++i)
            ana(i) = m0(i)*e1 + d0(i)*(e2 - e1)/denom;

        ana -= m;
        actual_error = ana.Norml2();

        if (fabs(actual_error - target_error) / target_error < 0.05)
            break;

        if (actual_error > target_error)
            dt_high = dt;
        else
            dt_low = dt;

        dt = sqrt(dt_low * dt_high);
    }

    if (fabs(actual_error - target_error) / target_error > 0.05){
        cout<<"Maximum iteration number achieved for tau="<<tau<<endl;
    } else {
        fout << tau << " " << dt << " " << oper.GetCounter() << " " << actual_error << "\n";
    }
}

fout.close();

}
