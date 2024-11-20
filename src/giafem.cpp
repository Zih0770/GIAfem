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


	DenseMatrix points(mesh.Dimension(), num_samples + 1);
        for (int i = 0; i <= num_samples; i++)
        {
            double radius = min_radius + i * (max_radius - min_radius) / num_samples;
            points(0, i) = radius;
            if (mesh.Dimension() > 1) points(1, i) = 0.0;
            if (mesh.Dimension() > 2) points(2, i) = 0.0; 
	    
	    //int elem_id;
	    //IntegrationPoint ips;
            //mesh.FindPoints(point, elem_id, ips);
            //double value = 0.0;
            //if (elem_id >= 0)
            //{
            //    ElementTransformation *trans = mesh.GetElementTransformation(elem_id);
            //    IntegrationPoint ip;
            //    if (trans->TransformBack(point, ip))
            //    {
            //        value = solution.GetValue(*trans, ip);
            //        std::cout << "Point [" << point(0) << ", " << point(1) << ", " << point(2)
            //                  << "] has function value " << value << " in element " << elem_id << ".\n";
            //    }
            //}
            //else
            //{
            //    std::cerr << "Warning: Point [" << point(0) << ", " << point(1) << ", " << point(2)
            //              << "] is outside the mesh.\n";
            //}

            //values.push_back(value);
            //radii.push_back(radius);
	}
        Array<int> elem_ids(num_samples + 1);
        Array<IntegrationPoint> ips(num_samples + 1);

        mesh.FindPoints(points, elem_ids, ips);
        for (int i = 0; i <= num_samples; i++)
        {
            double value = 0.0;

            int elem_id = elem_ids[i];
            if (elem_id >= 0) // Check if the point was found
            {
                ElementTransformation *trans = mesh.GetElementTransformation(elem_id);
                IntegrationPoint ip;
                ip.x = ips[i].x; // For 2D
                ip.y = ips[i].y;
                if (mesh.Dimension() == 3)
                {
                    ip.z = ips[i].z; // For 3D
                }

                value = solution.GetValue(*trans, ip);
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

} // namespace giafem

