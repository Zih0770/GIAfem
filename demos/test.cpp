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
#include "mfem.hpp"
#include "giafem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace mfem;
using namespace giafem;

class RelaxOp : public TimeDependentOperator
{
private:
    real_t tau, alpha;
    Vector d0_true;
public:
    RelaxOp(real_t tau_, real_t alpha_, const Vector &d0)
      : TimeDependentOperator(d0.Size()), tau(tau_), alpha(alpha_), d0_true(d0) { }

    // computes dm_dt = (d(t)-m)/tau
    void Mult(const Vector &m, Vector &dm_dt) const override
    {
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
        real_t t = this->GetTime() + dt;
        int n = m_old.Size();

        for (int i = 0; i < n; i++)
        {
            real_t d_next = d0_true(i) * exp(-alpha * t);
            dm_dt(i) = (d_next - m_old(i)) / tau / (1.0 + dt / tau);
        }
    }
};

// analytic solution for m0=initial m, forcing D0 e^{-αt}
void ComputeAnalytic(const Vector &m0,
                     Vector &m_ana,
                     double tau,
                     double alpha,
                     const Vector &D0,
                     double t)
{
    int n = m0.Size();
    double e1 = exp(-t/tau),
           e2 = exp(-alpha * t),
           denom = 1.0 - alpha*tau;
    for (int i = 0; i < n; i++)
    {
        m_ana(i) = m0(i)*e1
                  + D0(i)*(e2 - e1)/denom;
    }
}

int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/examples/star.mesh";
    const char *output_file = "error.dat";
    real_t tau     = 5.0;
    real_t dt      = 0.01;
    real_t t_final = 5.0;
    int solver_type = 3; // 0:FE,1:RK2,2:RK3,3:RK4,…
    real_t alpha = 0.5;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file.");
    args.AddOption(&output_file, "-f", "--file",
                   "Output file.");
    args.AddOption(&tau,       "-tau", "--tau",
                   "Relaxation time constant τ.");
    args.AddOption(&dt,        "-dt",  "--time-step",
                   "Time step size Δt.");
    args.AddOption(&t_final,   "-tf",  "--final-time",
                   "Final time T.");
    args.AddOption(&solver_type, "-s","--solver",
                   "Integrator: 0=FE, 1=RK2, 2=RK3, 3=RK4, …");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);


    Mesh mesh(mesh_file, 1, 1);
    int order = 2;
    H1_FECollection fec(order, mesh.Dimension());
    FiniteElementSpace fes(&mesh, &fec, 5);
    int size = fes.GetTrueVSize();
    Vector m(size), m0(size), d0(size);
    // Initial condition
    GridFunction m0_gf(&fes);
    VectorFunctionCoefficient m0_coeff(5,
      [&](const Vector &x, Vector &v)
      {
        v(0) = sin(M_PI*x[0])*sin(M_PI*x[1])*sin(M_PI*x[2]);
        v(1) = cos(M_PI*x[0])*sin(M_PI*x[1])*sin(M_PI*x[2]);
        v(2) = sin(M_PI*x[0])*cos(M_PI*x[1])*sin(M_PI*x[2]);
        v(3) = sin(M_PI*x[0])*sin(M_PI*x[1])*cos(M_PI*x[2]);
        v(4) = cos(M_PI*x[0])*cos(M_PI*x[1])*cos(M_PI*x[2]);
      });
    m0_gf.ProjectCoefficient(m0_coeff);
    m0_gf.GetTrueDofs(m0);
    m = m0; 
    
    //Project the forcing d0(x) = (j+1)*sin(πx)sin(πy)sin(πz)
    GridFunction d0_gf(&fes);
    VectorFunctionCoefficient d0_coeff(5,
      [&](const Vector &x, Vector &v)
      {
        double S = sin(M_PI*x[0])*sin(M_PI*x[1])*sin(M_PI*x[2]);
        for (int j = 0; j < 5; ++j) { v(j) = (j+1)*S; }
      });
    d0_gf.ProjectCoefficient(d0_coeff);
    d0_gf.GetTrueDofs(d0);


    RelaxOp oper(tau, alpha, d0);
    //auto ode_solver = unique_ptr<ODESolver>(ODESolver::Select(solver_type));
    //ODESolver *ode_solver = new ForwardEulerSolver();
    //ODESolver *ode_solver = new RK2Solver;
    //ODESolver *ode_solver = new RK3SSPSolver;
    //ODESolver *ode_solver = new RK4Solver;
    //ODESolver *ode_solver = new RK6Solver;
    //ODESolver *ode_solver = new AB1Solver;
    //ODESolver *ode_solver = new AB2Solver;
    //ODESolver *ode_solver = new AB3Solver;
    //ODESolver *ode_solver = new AB4Solver;
    //ODESolver *ode_solver = new AB5Solver;
    //ODESolver *ode_solver = new BaileySolver_test;

    //ODESolver *ode_solver = new BackwardEulerSolver;
    //ODESolver *ode_solver = new SDIRK23Solver(2);
    //ODESolver *ode_solver = new SDIRK33Solver;
    ODESolver *ode_solver = new ImplicitMidpointSolver;
    //ODESolver *ode_solver = new SDIRK23Solver;
    //ODESolver *ode_solver = new SDIRK34Solver;
    //ODESolver *ode_solver = new AM2Solver;
    ode_solver->Init(oper);

    ofstream fout(output_file);
    fout << "# t   L2_error\n";
    GridFunction m_gf(&fes), ana_gf(&fes), diff_gf(&fes);
    m_gf.SetFromTrueDofs(m);

    real_t t = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    while (t < t_final - 1e-12)
    {
        Vector ana_true(size);
        real_t e1 = exp(-t/tau), e2 = exp(-alpha*t), denom = 1.0 - alpha*tau;
        for (int i = 0; i < size; ++i)
        {
            ana_true(i) = m0(i)*e1 + d0(i)*(e2 - e1)/denom;
        }
        ana_gf.SetFromTrueDofs(ana_true);

        // compute L2 error
        diff_gf = m_gf;
        diff_gf -= ana_gf;
        real_t err = diff_gf.Norml2();
        auto current = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current - start;
        fout << t << " " << elapsed.count() << " " << err << "\n";

        real_t h = min(dt, t_final - t);
        ode_solver->Step(m, t, h);
        m_gf.SetFromTrueDofs(m);
    }
    fout.close();

    delete ode_solver;
    return 0;
}

