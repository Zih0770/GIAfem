// giafem.cpp
#include "giafem.hpp"
#include <cmath>
#include <fstream>

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

        for (int i = 0; i <= num_samples; i++)
        {
            double radius = min_radius + i * (max_radius - min_radius) / num_samples;
	    Vector point(3);
            point(0) = radius;
            point(1) = 0.0;
            point(2) = 0.0;

	    bool point_found = false;
            double value = 0.0;
            for (int elem_id = 0; elem_id < mesh.GetNE(); elem_id++)  // Iterating over each element
            {
                ElementTransformation *trans = mesh.GetElementTransformation(elem_id);

                // Try to map 'point' back to reference space within this element
                IntegrationPoint ip;
                trans->TransformBack(point, ip);
                
		bool inside = true;
                if (mesh.Dimension() >= 1 && (ip.x < 0.0 || ip.x > 1.0)) inside = false;
                if (mesh.Dimension() >= 2 && (ip.y < 0.0 || ip.y > 1.0)) inside = false;
                if (mesh.Dimension() == 3 && (ip.z < 0.0 || ip.z > 1.0)) inside = false;
                
		if (inside)
                {
                    value = solution.GetValue(*trans, ip);  // Evaluate GridFunction at this point
                    point_found = true;
                    break;
                }
            }

            if (!point_found)
            {
                // Handle case if point is not found in any element
                value = 0.0;  // Assign a default or handle as needed
            }

        values.push_back(value);
        radii.push_back(radius);

        }
    }

    void plot::SaveRadialSolution(const std::string &filename, int num_samples, double min_radius, double max_radius)
    {
        std::vector<double> radii;
        std::vector<double> values;
        EvaluateRadialSolution(num_samples, min_radius, max_radius, radii, values);

        std::ofstream ofs(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open the file " << filename << " for writing.\n";
            return;
        }
	
	ofs << "# Radius Solution\n";
        for (size_t i = 0; i < radii.size(); i++)
        {
            ofs << radii[i] << " " << values[i] << "\n";
        }
        ofs.close();
    }

} // namespace giafem

