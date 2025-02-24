#include <gmsh.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

void createConcentricSphericalLayers(const std::vector<double> &radii, double meshSizeMin, double meshSizeMax, const std::string &outputFileName) {
    int numLayers = radii.size();
    
    if (numLayers < 1) {
        std::cerr << "Error: There should be at least one layer." << std::endl;
        return;
    }
    // Initialize Gmsh
    gmsh::initialize();
    gmsh::model::add("ConcentricSphericalLayers");

    // Set mesh size options
    gmsh::option::setNumber("Mesh.MeshSizeMin", meshSizeMin);
    gmsh::option::setNumber("Mesh.MeshSizeMax", meshSizeMax);

    // Tagging scheme starting from 301
    int layerTag = 1;
    int surfaceTag = 1;
    for (int i = 0; i < numLayers; ++i) {
        gmsh::model::occ::addSphere(0, 0, 0, radii[i]);
    } 
    gmsh::model::occ::synchronize();
    gmsh::model::addPhysicalGroup(3, {1}, layerTag);
    gmsh::model::setPhysicalName(3, layerTag, "sphericalLayer_1");
    std::vector<std::pair<int, int>> surfaceEntities;
    gmsh::model::getBoundary({{3, 1}}, surfaceEntities, false, false, false);
    std::pair<int, int> surface = surfaceEntities[0];
    if (surface.first == 2) {
        gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
        gmsh::model::setPhysicalName(2, surfaceTag,
                                     "sphericalSurface_" + std::to_string(1));
    }
    for (int i = 1; i < numLayers; ++i) {
        std::vector<std::pair<int, int> > ov;
        std::vector<std::vector<std::pair<int, int> > > ovv;

        gmsh::model::occ::cut({{3, i+1}}, {{3, i}},  ov, ovv, -1, false, false); 
	gmsh::model::occ::synchronize();

	std::vector<int> volumeTags;
        for (const auto &entity : ov) {
            volumeTags.push_back(entity.second);  // Extract only the tag part
	}
        ++layerTag;
        gmsh::model::addPhysicalGroup(3, volumeTags, layerTag);
        gmsh::model::setPhysicalName(3, layerTag, "sphericalLayer_" + std::to_string(i + 1));
        for (const auto &volumeTag : volumeTags) {
            std::vector<std::pair<int, int>> surfaceEntities;
            gmsh::model::getBoundary({{3, volumeTag}}, surfaceEntities, false, false, false);	
	    std::pair<int, int> surface = surfaceEntities[0];
            if (surface.first == 2) {
                ++surfaceTag;
                gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
                gmsh::model::setPhysicalName(2, surfaceTag, 
                                             "sphericalSurface_" + std::to_string(i + 1));
            }
	}
    }
    for (int i = 1; i < numLayers; ++i) {
        gmsh::model::occ::remove({{3, i+1}});
    }
    gmsh::model::occ::synchronize();

    // Generate 3D mesh
    gmsh::model::mesh::generate(3);

    // Save the mesh
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write(outputFileName);

    // Finalize Gmsh
    gmsh::finalize();
}

std::vector<double> parseRadii(const std::string &radiiStr) {
    std::vector<double> radii;
    std::istringstream iss(radiiStr);
    std::string token;

    while (std::getline(iss, token, '-')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        radii.push_back(std::stod(token));
    }

    return radii;
}

int main(int argc, char **argv) {

    std::vector<double> radii = {6.38, 8.0};
    double meshSizeMin = 1.0;
    double meshSizeMax = 1.0;
    std::string outputFileName = "concentric_spherical_layers.msh";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-r" && i + 1 < argc) {
            radii = parseRadii(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            std::string meshSizeStr = argv[++i];
            auto meshSizes = parseRadii(meshSizeStr); 
            if (meshSizes.size() == 2) {
                meshSizeMin = meshSizes[0];
                meshSizeMax = meshSizes[1];
            } else {
                std::cerr << "Error: mesh sizes should have two values.\n";
                return 1;
            }
        } else if (arg == "-o" && i + 1 < argc) {
            outputFileName = argv[++i];  // Set the output file name
        }
    }

    // Run the spherical layers creation function
    createConcentricSphericalLayers(radii, meshSizeMin, meshSizeMax, outputFileName);

    return 0;
}

