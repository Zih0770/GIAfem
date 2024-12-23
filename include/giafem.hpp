#ifndef GIAFEM_HPP
#define GIAFEM_HPP

#include "mfem.hpp"  // Include MFEM for GridFunction, Mesh, etc.
#include <vector>
#include <string>
#include <utility>

namespace giafem
{
    using namespace mfem;

    class plot
    {
    public:
        // Constructor to initialize with solution and mesh
        plot(const GridFunction &solution, Mesh &mesh);

        // Subroutine to evaluate the radial solution at evenly spaced points
        void EvaluateRadialSolution(int num_samples, double min_radius, double max_radius, std::vector<double> &radii, 
                                    std::vector<double> &values);

        // Optional subroutine to save the evaluated radial solution data to a file
        void SaveRadialSolution(const std::string &filename, int num_samples, double min_radius, double max_radius);

    private:
        const GridFunction &solution;  // Reference to the solution GridFunction
        Mesh &mesh;              // Reference to the mesh
    };


    class parse
    {
    public:
        // Function to parse 1D properties from a file with dynamic columns
        std::vector<std::vector<double>> properties_1d(const std::string &filename);
    };


    class interp
    {
    public:
        // Function to interpolate a 1D property from dynamic data
        mfem::Array<mfem::FunctionCoefficient *>
        PWCoef_1D(const std::vector<std::pair<double, double>> &radius_property,
                                 int num_attributes,
                                 const std::string &method = "linear");
    };

} // namespace giafem

#endif // GIAFEM_HPP

