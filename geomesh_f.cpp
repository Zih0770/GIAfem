#include <gmsh.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>


std::vector<double> extractLayerBoundaries(const std::string &fileName) {
    std::vector<double> radii;
    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << fileName << std::endl;
        return radii;
    }

    std::string line;
    double previousRadius = -1.0;

    int lineCount = 0;
    while (std::getline(file, line)) {
        if (lineCount < 3) {
            lineCount++;
            continue;
        }
	    
	std::istringstream iss(line);
        double radius, density, pWave, sWave, bulkM, shearM;
        if (iss >> radius >> density >> pWave >> sWave >> bulkM >> shearM) {
            if (std::abs(radius - previousRadius) < 1e-6) {
                radii.push_back(radius / 1e6);
            }
            previousRadius = radius;
        }
    }
    radii.push_back(previousRadius / 1e6);
    radii.push_back(10.0);

    file.close();
    return radii;
}

struct PropertyData {
    double radius;
    double density;
    double pWaveSpeed;
    double sWaveSpeed;
    double bm;
    double sm;
};

std::vector<PropertyData> parsePropertyData(const std::string &fileName) {
    std::vector<PropertyData> data;
    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << fileName << std::endl;
        return data;
    }

    std::string line;
    int lineCount = 0;

    while (std::getline(file, line)) {
        if (lineCount < 3) {
            lineCount++;
            continue;
        }

        std::istringstream iss(line);
        double radius, density, pWaveSpeed, sWaveSpeed, bulkQuality, shearQuality;
        if (iss >> radius >> density >> pWaveSpeed >> sWaveSpeed >> bulkQuality >> shearQuality) {
            data.push_back({
                radius / 1e6,
                density,
                pWaveSpeed,
                sWaveSpeed,
                bulkQuality,
                shearQuality
            });
        }
    }

    file.close();
    return data;
}

double interpolateProperty(double r, const std::vector<PropertyData> &data, const std::function<double(const PropertyData &)> &propertyExtractor) {
    for (size_t i = 0; i < data.size() - 1; ++i) {
        if (r >= data[i].radius && r <= data[i + 1].radius) {
            double r1 = data[i].radius, r2 = data[i + 1].radius;
            double p1 = propertyExtractor(data[i]), p2 = propertyExtractor(data[i + 1]);
            return p1 + (p2 - p1) * (r - r1) / (r2 - r1);
        }
    }

    // If outside the range, return the nearest value
    if (r < data.front().radius) return propertyExtractor(data.front());
    return propertyExtractor(data.back());
}



void createConcentricSphericalLayers(const std::vector<double> &radii, const std::vector<PropertyData> &propertyData, double meshSizeMin, double meshSizeMax, const std::string &outputFileName) {
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
        gmsh::model::setPhysicalName(2, surfaceTag, "sphericalSurface_1");
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

    std::vector<std::size_t> nodeTags;
    std::vector<double> nodeCoords;
    std::vector<std::vector<double>> densityField, pWaveField, sWaveField, bmField, smField;

    std::vector<double> parametricCoord;
    gmsh::model::mesh::getNodes(nodeTags, nodeCoords, parametricCoord, -1, -1, false, true);

    for (size_t i = 0; i < nodeTags.size(); ++i) {
        double x = nodeCoords[3 * i];
        double y = nodeCoords[3 * i + 1];
        double z = nodeCoords[3 * i + 2];
        double r = std::sqrt(x * x + y * y + z * z);  // Radial distance

        auto Density = std::vector<double>{
	    interpolateProperty(r, propertyData, [](const PropertyData &d) { return d.density; })
	};
	auto BM = std::vector<double>{
            interpolateProperty(r, propertyData, [](const PropertyData &d) { return d.bm; })
        };
	auto SM = std::vector<double>{
            interpolateProperty(r, propertyData, [](const PropertyData &d) { return d.sm; })
        };
        densityField.push_back(Density);
	bmField.push_back(BM);
	smField.push_back(SM);
    }


    std::cout << "Size of nodeTags: " << nodeTags.size() << std::endl;
    std::cout << "Size of densityField: " << densityField.size() << std::endl;

    //std::vector<std::vector<double>> densityFieldWrapped(1, densityField);

    gmsh::view::add("Density");
    gmsh::view::addModelData(1, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, densityField);
    gmsh::view::write(1, outputFileName + "_density.pos");

    //gmsh::view::add("P-Wave Speed");
    //gmsh::view::addModelData(1, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, {pWaveField});
    //gmsh::view::write(1, "pwave_distribution.pos");

    //gmsh::view::add("S-Wave Speed");
    //gmsh::view::addModelData(2, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, {sWaveField});
    //gmsh::view::write(2, "swave_distribution.pos");

    gmsh::view::add("Bulk Modulus");
    gmsh::view::addModelData(2, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, bmField);
    gmsh::view::write(2, "bm_distribution.pos");

    gmsh::view::add("Shear Modulus");
    gmsh::view::addModelData(3, 0, "ConcentricSphericalLayers", "NodeData", nodeTags, smField);
    gmsh::view::write(3, "sm_distribution.pos");

    // Save the mesh
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write(outputFileName + ".msh");

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
    double meshSizeMin = 1.0;
    double meshSizeMax = 1.0;
    std::string inputFileName = "prem.200.noiso";
    std::string outputFileName = "concentric_spherical_layers.msh";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-i" && i + 1 < argc) {
            inputFileName = argv[++i];
        }else if (arg == "-s" && i + 1 < argc) {
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
            outputFileName = argv[++i];
        }
    }

    std::vector<double> radii = extractLayerBoundaries(inputFileName);

    std::cout << "Detected layer radii (Mm): ";
    for (const double r : radii) {
        std::cout << std::fixed << std::setprecision(2) << r << " ";
    }
    std::cout << std::endl;

    std::vector<PropertyData> propertyData = parsePropertyData(inputFileName);

    // Run the spherical layers creation function
    createConcentricSphericalLayers(radii, propertyData, meshSizeMin, meshSizeMax, outputFileName);

    return 0;
}

