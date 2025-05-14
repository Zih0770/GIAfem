#include <gmsh.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>


void createConcentricSphericalLayers(const std::vector<double> &radii, double meshSizeMin, double meshSizeMax, int elementOrder, const std::string &outputFileName) {
    int numLayers = radii.size();
    
    if (numLayers < 1) {
        std::cerr << "Error: There should be at least one layer." << std::endl;
        return;
    }

    gmsh::initialize();
    gmsh::model::add("ConcentricSphericalLayers");

    // Set mesh size options
    gmsh::option::setNumber("Mesh.MeshSizeMin", meshSizeMin);
    gmsh::option::setNumber("Mesh.MeshSizeMax", meshSizeMax);

    for (int i = 0; i < numLayers; ++i) {
        gmsh::model::occ::addSphere(0, 0, 0, radii[i]);
    }
    gmsh::model::occ::synchronize();
    //Innest layer
    int layerTag = 1;
    int surfaceTag = 1;
    gmsh::model::addPhysicalGroup(3, {1}, layerTag);
    gmsh::model::setPhysicalName(3, layerTag, "layer_1");
    std::vector<std::pair<int, int>> surfaceEntities;
    gmsh::model::getBoundary({{3, 1}}, surfaceEntities, false, false, false); //combined - oriented - recursive
    std::pair<int, int> surface = surfaceEntities[0];
    if (surface.first == 2) {
        gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
        gmsh::model::setPhysicalName(2, surfaceTag, "surface_" + std::to_string(1));
    }
    //Other layers
    for (int i = 1; i < numLayers; ++i) {
        std::vector<std::pair<int, int>> ov;
        std::vector<std::vector<std::pair<int, int>>> ovv;

        gmsh::model::occ::cut({{3, i+1}}, {{3, i}},  ov, ovv, -1, false, false); //auto-assigns tags - removeObject - removeTool 
	    gmsh::model::occ::synchronize();

	    std::vector<int> volumeTags;
        for (const auto &entity : ov) {
            volumeTags.push_back(entity.second);
	    }
        ++layerTag;
        gmsh::model::addPhysicalGroup(3, volumeTags, layerTag);
        gmsh::model::setPhysicalName(3, layerTag, "layer_" + std::to_string(i+1));
        for (const auto &volumeTag : volumeTags) {
            std::vector<std::pair<int, int>> surfaceEntities;
            gmsh::model::getBoundary({{3, volumeTag}}, surfaceEntities, false, false, false);	
	        std::pair<int, int> surface = surfaceEntities[0]; //Only take the inner surface
            if (surface.first == 2) {
                ++surfaceTag;
                gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
                gmsh::model::setPhysicalName(2, surfaceTag, "surface_" + std::to_string(i+1));
            }
	    }
    }
    for (int i = 1; i < numLayers; ++i) {
        gmsh::model::occ::remove({{3, i+1}});
    }
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(3);

    gmsh::option::setNumber("Mesh.Algorithm3D", 10); //1-Delaunay, 4-Frontal, 7-MMG3D, 9-R-tree Delaunay, 10-HXT (Frontal-Delaunay), 11-Automatic
    gmsh::option::setNumber("Mesh.Optimize", 3);
    gmsh::option::setNumber("Mesh.OptimizeNetgen", 1);
    gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
    //gmsh::option::setNumber("Mesh.SecondOrderIncomplete", 0);
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write(outputFileName);

    gmsh::finalize();
}

std::vector<double> parseString(const std::string &string_arg) {
    std::vector<double> entries;
    std::istringstream iss(string_arg);
    std::string token;

    while (std::getline(iss, token, '-')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        entries.push_back(std::stod(token));
    }

    return entries;
}


int main(int argc, char **argv) {

    std::vector<double> radii = {6380.0, 8000.0};
    double meshSizeMin = 20.0;
    double meshSizeMax = 200.0;
    int elementOrder = 1;
    std::string outputFileName = "concentric_spherical_layers.msh";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-r" && i + 1 < argc) {
            radii = parseString(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            std::string meshSizeStr = argv[++i];
            auto meshSizes = parseString(meshSizeStr); 
            if (meshSizes.size() == 2) {
                meshSizeMin = meshSizes[0];
                meshSizeMax = meshSizes[1];
            } else {
                std::cerr << "Error: mesh sizes should have two values.\n";
                return 1;
            }
        } else if (arg == "-o" && i + 1 < argc) {
            elementOrder = std::stoi(argv[++i]);
        } else if (arg == "-o" && i + 1 < argc) {
            outputFileName = argv[++i];
        }
    }

    createConcentricSphericalLayers(radii, meshSizeMin, meshSizeMax, elementOrder, outputFileName);

    return 0;
}

