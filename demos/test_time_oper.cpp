#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

using namespace mfem;
using namespace std;
namespace fs = std::filesystem;


class mEvolutionOperator : public TimeDependentOperator
{
private:
    real_t tau;
    FiniteElementSpace &fes;
    mutable GridFunction d_gf;

public:
    mEvolutionOperator(FiniteElementSpace &fes_, double tau_)
        : TimeDependentOperator(fes_.GetTrueVSize()), fes(fes_),
          d_gf(&fes_), tau(tau_) {}

    virtual void Mult(const Vector &m_vec, Vector &dm_dt_vec) const override
    {
        real_t current_time = this->GetTime();

        VectorFunctionCoefficient d_coeff(5, 
            [current_time](const Vector &x, Vector &d) {
                d.SetSize(5);
                d[0] = sin(current_time) * cos(x[0]);  
                d[1] = cos(current_time) * sin(x[1]);  
                d[2] = 0.5 * sin(current_time + x[2]); 
                d[3] = 0.5 * cos(current_time + x[0]); 
                d[4] = 0.3 * sin(current_time + x[1]); 
            });

        d_gf.ProjectCoefficient(d_coeff);

        Vector d_m(fes.GetTrueVSize());
        d_gf.GetTrueDofs(d_m);

        for (int i = 0; i < d_m.Size(); i++)
        {
            dm_dt_vec[i] = (d_m[i] - m_vec[i]) / tau;
        }
    }
};

int main()
{
    string base_dir = "data";
    string output_dir = base_dir + "/m_output";

    if (!fs::exists(base_dir))
    {
        fs::create_directory(base_dir);
        cout << "Created directory: " << base_dir << endl;
    }
    if (!fs::exists(output_dir))
    {
        fs::create_directory(output_dir);
        cout << "Created directory: " << output_dir << endl;
    }

    // Simulation parameters
    real_t tau = 1.0;
    real_t t_final = 10.0;
    real_t dt = 0.1;

    Mesh mesh = Mesh::MakeCartesian3D(16, 16, 16, Element::HEXAHEDRON);
    H1_FECollection fec(2, mesh.Dimension());
    FiniteElementSpace fes(&mesh, &fec, 5);

    GridFunction m_gf(&fes);
    m_gf = 0.0;

    Vector m_vec;
    m_gf.GetTrueDofs(m_vec);

    mEvolutionOperator oper(fes, tau);

    // Choose an ODE solver (Runge-Kutta 4)
    RK4Solver ode_solver;
    ode_solver.Init(oper);

    // Time-stepping loop
    real_t t = 0.0;
    int step = 0;
    while (t < t_final)
    {
        real_t dt_real = min(dt, t_final - t);
        ode_solver.Step(m_vec, t, dt_real);

        // Convert m Vector to GridFunction
        m_gf.SetFromTrueDofs(m_vec);

        string filename = output_dir + "/m_step_" + to_string(step) + ".gf";
        ofstream out(filename);
        out.precision(8);
        m_gf.Save(out);
        cout << "Saved: " << filename << endl;

        step++;
    }

    string mesh_filename = output_dir + "/mesh.mesh";
    ofstream mesh_out(mesh_filename);
    mesh_out.precision(8);
    mesh.Print(mesh_out);
    cout << "Saved mesh: " << mesh_filename << endl;

    return 0;
}

