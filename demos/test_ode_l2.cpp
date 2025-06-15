#include "mfem.hpp"
#include "giafem.hpp" 
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>
using namespace std;
using namespace mfem;
using namespace giafem;

//-----------------------------------------------------------------------------
// Spatial patterns:
//   d₀ⱼ(x) = (j+1) * sin(2πx) * cos(2πy) * sin(2πz)
//   m₀ⱼ(x) = (j+1) * cos(πx)   * sin(πy) * cos(πz)
//
// Analytic solution per DOF:
//   m_ana(t) = m₀ e^(−t/τ)
//            + d₀ [ sin(ωt) − ωτ cos(ωt) + ωτ e^(−t/τ) ] / (1 + (ωτ)²)
//-----------------------------------------------------------------------------
//   m_ana(t) = m0 e^(−t/τ)
//            + d0 [ cos(ωt) + ωτ sin(ωt) − e^(−t/τ) ]
//                  / (1 + (ωτ)²)
class CountingRelaxOp : public TimeDependentOperator
{
private:
    real_t tau, omega;
    Vector d0_true;
    mutable int mult_calls;
public:
    CountingRelaxOp(real_t tau_, real_t omega_, const Vector &d0)
      : TimeDependentOperator(d0.Size()),
        tau(tau_), omega(omega_), d0_true(d0), mult_calls(0) { }

    void ResetCounter() const { mult_calls = 0; }
    int  GetCounter()   const { return mult_calls; }

    void Mult(const Vector &m, Vector &dm_dt) const override
    {
        mult_calls++;
        real_t t = this->GetTime();
        for (int i = 0; i < m.Size(); i++)
        {
            real_t forcing = d0_true(i) * cos(omega * t);
            dm_dt(i) = (forcing - m(i)) / tau;
        }
    }

    void ImplicitSolve(const real_t dt,
                       const Vector &m_old,
                       Vector &dm_dt) override
    {
        mult_calls++;
        real_t t1 = this->GetTime() + dt;
        for (int i = 0; i < m_old.Size(); i++)
        {
            real_t forcing = d0_true(i) * cos(omega * t1);
            dm_dt(i) = (forcing - m_old(i)) / (tau * (1.0 + dt / tau));
        }
    }
};

static void ComputeAnalytic(const Vector &m0,
                            const Vector &d0,
                            real_t tau, real_t omega,
                            real_t t, Vector &m_ana)
{
    const real_t e1 = exp(-t/tau);
    const real_t denom = 1.0 + omega*omega*tau*tau;
    for (int i = 0; i < m0.Size(); i++)
    {
        real_t A = ( cos(omega*t)
                + omega*tau*sin(omega*t)
                - e1 )
            / denom;
        m_ana(i) = m0(i)*e1 + d0(i)*A;
    }
}

int main()
{
    Mesh mesh("mesh/examples/beam-tet.mesh",1,1);
    H1_FECollection fec(2,mesh.Dimension());
    FiniteElementSpace fes(&mesh,&fec,5);
    int sz = fes.GetTrueVSize();

    Vector d0(sz), m0(sz), m(sz);
    {
        GridFunction d_gf(&fes), m_gf(&fes);
        VectorFunctionCoefficient d_coef(5,
          [](auto &x, auto &v){
            real_t s1=sin(2*M_PI*x[0]), c2=cos(2*M_PI*x[1]), s3=sin(2*M_PI*x[2]);
            for(int j=0;j<5;j++) v(j)=(j+1)*s1*c2*s3;
          });
        d_gf.ProjectCoefficient(d_coef); d_gf.GetTrueDofs(d0);
        VectorFunctionCoefficient m_coef(5,
          [](auto &x, auto &v){
            real_t c1=cos(M_PI*x[0]), s2=sin(M_PI*x[1]), c3=cos(M_PI*x[2]);
            for(int j=0;j<5;j++) v(j)=(j+1)*c1*s2*c3;
          });
        m_gf.ProjectCoefficient(m_coef); m_gf.GetTrueDofs(m0);
        m = m0;
    }

    const vector<int> solvers = {99, 1, 2, 3, 4, 21}; // FE, RK2, RK3, Bailey
    //const vector<int> solvers = {4};
    vector<real_t> l2s;
    for(real_t l2=1e-6; l2<=1e-2; l2*=sqrt(10)) l2s.push_back(l2);
    //vector<real_t> omega_t = {0.1, 0.316228, 1.0};

    const real_t Tfinal = 100.0;

    for(int st: solvers)
    {
        cout<<"Solver "<<to_string(st)<<": "<<endl;
        auto ode = (st!=99 ? ODESolver::Select(st)
                          : make_unique<BaileySolver_test>());

        //ofstream fout("solver_"+to_string(st)+"_frequency_"+to_string(eps)+".dat");
        ofstream fout("solver_"+to_string(st)+"_l2_1e8.dat");
        fout<<"#omega_tau  dt  calls  L2err\n";
        for(real_t eps: l2s)
        {
            real_t tau=1.0, omega=0.01;
            cout<<"target l2 = "<<eps<<":"<<endl;
            CountingRelaxOp op(tau,omega,d0);
            ode->Init(op);

            real_t lo=1e-4, hi=10.0, dt=0.1, err=0;
            int calls=0;
            bool max_iter = false;
            for(int it=0; it<18; it++)
            {
                cout<<"dt = "<<dt<<endl;
                op.ResetCounter(); m=m0; real_t t=0;
                while(t<Tfinal-1e-8)
                {
                    real_t h=min(dt,Tfinal-t);
                    ode->Step(m,t,h);
                }
                Vector ma(sz);
                ComputeAnalytic(m0,d0,tau,omega,t,ma);
                m-=ma; err=m.Norml2()/ma.Norml2(); calls=op.GetCounter();
                if (st == 99) cout<<"Bailey l2 error: "<<err<<endl;

                if(fabs(err-eps)/eps<0.05) {
                    break;
                } else {
                    if(err>eps) hi=dt; else lo=dt;
                }

                dt=sqrt(lo*hi);
                if (it==17) {
                    cerr<<"Max iter!"<<endl;
                    max_iter = true;
                }
            }
            if (!max_iter)
                fout<<eps<<"  "<<dt<<"  "<<calls<<"  "<<err<<"\n";
        }
    }
    return 0;
}
