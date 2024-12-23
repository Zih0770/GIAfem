// giafem.cpp
#include "giafem.hpp"
#include <cmath>
#include <functional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace giafem
{
    plot::plot(const GridFunction &sol, Mesh &msh)
        : solution(sol), mesh(msh) {}

    void plot::EvaluateRadialSolution(int num_samples, double min_radius, double max_radius, 
		                        std::vector<double> &radii, std::vector<double> &values)
    {
        double mesh_max_radius = 0.0;
        
	for (int i = 0; i < mesh.GetNV(); i++)
        {
            Vector node;
            //mesh.GetVertex(i, node);
            const double* vertex_coords = mesh.GetVertex(i);
            node.SetSize(mesh.Dimension());
            for (int d = 0; d < mesh.Dimension(); d++)
            {
                node(d) = vertex_coords[d];
            }

	    double radius = node.Norml2();
            if (radius > mesh_max_radius)
                mesh_max_radius = radius;
        }
	max_radius = std::min(max_radius, mesh_max_radius);


	DenseMatrix points(mesh.Dimension(), num_samples);
        for (int i = 0; i < num_samples; i++)
        {
            double radius = min_radius + i * (max_radius - min_radius) / num_samples;
            points(0, i) = radius;
            if (mesh.Dimension() > 1) points(1, i) = 0.0;
            if (mesh.Dimension() > 2) points(2, i) = 0.0; 
	}
       

        Array<int> elem_ids(num_samples);
	Array<IntegrationPoint> ips(num_samples);
        
        mesh.FindPoints(points, elem_ids, ips);
        
	for (int i = 0; i < num_samples; i++)
        {
            double value = 0.0;
            int elem_id = elem_ids[i];
            if (elem_id >= 0) // Check if the point was found
            {
                ElementTransformation *trans = mesh.GetElementTransformation(elem_id);
                
		value = solution.GetValue(*trans, ips[i]);
                std::cout << "Point (" << points(0, i) << ", "
                          << (mesh.Dimension() > 1 ? points(1, i) : 0.0) << ") "
                          << "has value " << value << " in element " << elem_id << ".\n";
            }
            else
            {
                std::cerr << "Warning: Point (" << points(0, i) << ", "
                          << (mesh.Dimension() > 1 ? points(1, i) : 0.0) << ") "
                          << "is outside the mesh.\n";
            }

            radii.push_back(points(0, i));
            values.push_back(value);
	}
    }

    void plot::SaveRadialSolution(const std::string &filename, int num_samples, double min_radius, double max_radius)
    {
        std::vector<double> radii;
        std::vector<double> values;
        EvaluateRadialSolution(num_samples, min_radius, max_radius, radii, values);

        std::ofstream ofs(filename);
        //if (!outfile.is_open()) {
        //    std::cerr << "Error: Could not open the file " << filename << " for writing.\n";
        //    return;
        //}
	
	ofs << filename<<"\n";
        for (size_t i = 0; i < radii.size(); i++)
        {
            ofs << radii[i] << " " << values[i] << "\n";
        }
        ofs.close();
    }

    

    std::vector<std::vector<double>> parse::properties_1d(const std::string &filename)
    {
        std::ifstream infile(filename);
        if (!infile.is_open())
        {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        std::vector<std::vector<double>> properties;
        std::string line;

        int num_columns = 0; // Local variable to track the number of columns

        std::getline(infile, line);
        std::getline(infile, line);
        std::getline(infile, line);

        // Parse the file
        while (std::getline(infile, line))
        {
            std::istringstream iss(line);
            std::vector<double> row;
            double value;

            // Read all columns in the row
            while (iss >> value)
            {
                row.push_back(value);
            }

            // Determine the number of columns from the first valid row
            if (properties.empty() && !row.empty())
            {
                num_columns = row.size();
            }

            // Ensure all rows have the same number of columns
            if (!row.empty() && row.size() != num_columns)
            {
                throw std::runtime_error("Inconsistent number of columns in file: " + filename);
            }

            if (!row.empty())
            {
                properties.push_back(row);
            }
        }

        infile.close();

        return properties;
    }


    mfem::Array<mfem::FunctionCoefficient *>
    interp::PWCoef_1D(const std::vector<std::pair<double, double>> &radius_property,
                                     int num_attributes,
                                     const std::string &method)
    {
        if (radius_property.empty())
        {
            throw std::runtime_error("Radius-property array is empty.");
        }

        // Define the interpolation function
        std::function<double(double)> interpolate;

        if (method == "linear")
        {
            interpolate = [radius_property](double radius) -> double {
                if (radius > radius_property.back().first)
                {
                    return 0.0; // Return zero outside the range
                }

                for (size_t i = 1; i < radius_property.size(); ++i)
                {
                    if (radius <= radius_property[i].first)
                    {
                        double r1 = radius_property[i - 1].first;
                        double r2 = radius_property[i].first;
                        double v1 = radius_property[i - 1].second;
                        double v2 = radius_property[i].second;

                        // Linear interpolation
                        return v1 + (v2 - v1) * (radius - r1) / (r2 - r1);
                    }
                }
                return 0.0; 
            };
        }
        else if (method == "spline")
        {
            throw std::runtime_error("Spline interpolation is not implemented yet.");
        }
        else
        {
            throw std::runtime_error("Unsupported interpolation method: " + method);
        }

        // Create FunctionCoefficients for each attribute
        mfem::Array<mfem::FunctionCoefficient *> coefficients;
        for (int i = 1; i <= num_attributes; ++i)
        {
            // Create a FunctionCoefficient for this attribute
            mfem::FunctionCoefficient *fc = new mfem::FunctionCoefficient([interpolate](const mfem::Vector &x) -> double {
                double r = sqrt(x[0] * x[0] + x[1] * x[1] + (x.Size() == 3 ? x[2] * x[2] : 0.0));
                return interpolate(r);
            });

            coefficients.Append(fc);
        }

        return coefficients;
    }



} // namespace giafem

