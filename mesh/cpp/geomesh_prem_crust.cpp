#include <gmsh.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <map>
#include <functional>
#include <stdexcept>

// ============================================================================
//                         PREM radii (original logic)
// ============================================================================
std::vector<double> extractLayerBoundaries(const std::string &fileName, double& R) {
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
            if (std::abs(radius - previousRadius) < 1e-3) {
                radii.push_back(radius);
            }
            previousRadius = radius;
        }
    }
    radii.push_back(previousRadius);
    R = radii.back();
    double fac = 1.2;
    double radius_max = fac * R;
    radii.push_back(radius_max);
    for (double& r : radii) {
        r /= R;
    }

    file.close();
    return radii;
}

// ============================================================================
//                         CRUST grid (original logic)
// ============================================================================
struct CrustGrid {
    int nlon = 0, nlat = 0;
    double lon0 = -180.0, lat0 = -90.0;
    double dlon = 1.0, dlat = 1.0;
    std::vector<double> d1; // nlat * nlon
    std::vector<double> d2;
    inline int idx(int i, int j) const { return j * nlon + i; }
};

static bool loadXYZ(const std::string &fname, std::vector<double> &lon, std::vector<double> &lat, std::vector<double> &val) {
    std::ifstream f(fname);
    if(!f) return false;
    lon.clear(); lat.clear(); val.clear();
    double a,b,c;
    while(f >> a >> b >> c) { lon.push_back(a); lat.push_back(b); val.push_back(c); }
    return !lon.empty();
}

static bool buildCrustGrid(const std::string &file1_d1, const std::string &file2_d2, CrustGrid &G) {
    std::vector<double> lon1, lat1, v1, lon2, lat2, v2;
    if(!loadXYZ(file1_d1, lon1, lat1, v1)) { std::cerr << "Error: cannot read " << file1_d1 << "\n"; return false; }
    if(!loadXYZ(file2_d2, lon2, lat2, v2)) { std::cerr << "Error: cannot read " << file2_d2 << "\n"; return false; }
    if(lon1.size() != lon2.size()) { std::cerr << "Error: CRUST xyz sizes differ.\n"; return false; }

    std::vector<double> ulon = lon1, ulat = lat1;
    std::sort(ulon.begin(), ulon.end()); ulon.erase(std::unique(ulon.begin(), ulon.end()), ulon.end());
    std::sort(ulat.begin(), ulat.end()); ulat.erase(std::unique(ulat.begin(), ulat.end()), ulat.end());

    G.nlon = (int)ulon.size(); G.nlat = (int)ulat.size();
    G.lon0 = ulon.front(); G.lat0 = ulat.front();
    G.dlon = (ulon.back() - ulon.front()) / (G.nlon - 1);
    G.dlat = (ulat.back() - ulat.front()) / (G.nlat - 1);

    G.d1.assign(G.nlon * G.nlat, 0.0);
    G.d2.assign(G.nlon * G.nlat, 0.0);

    auto normLon = [](double L){ double x = std::fmod(L + 180.0, 360.0); if(x < 0) x += 360.0; return x - 180.0; };

    std::map<std::pair<int,int>, int> ij;
    for(size_t k=0; k<lon1.size(); ++k) {
        double L = normLon(lon1[k]), B = std::max(-90.0, std::min(90.0, lat1[k]));
        int i = (int)std::llround((L - G.lon0) / G.dlon);
        int j = (int)std::llround((B - G.lat0) / G.dlat);
        if(i>=0 && i<G.nlon && j>=0 && j<G.nlat) ij[{i,j}] = (int)k;
    }
    for(int j=0; j<G.nlat; ++j) for(int i=0; i<G.nlon; ++i) {
        auto it = ij.find({i,j}); if(it==ij.end()) continue; int k = it->second; G.d1[G.idx(i,j)] = v1[(size_t)k];
    }
    ij.clear();
    for(size_t k=0; k<lon2.size(); ++k) {
        double L = normLon(lon2[k]), B = std::max(-90.0, std::min(90.0, lat2[k]));
        int i = (int)std::llround((L - G.lon0) / G.dlon);
        int j = (int)std::llround((B - G.lat0) / G.dlat);
        if(i>=0 && i<G.nlon && j>=0 && j<G.nlat) ij[{i,j}] = (int)k;
    }
    for(int j=0; j<G.nlat; ++j) for(int i=0; i<G.nlon; ++i) {
        auto it = ij.find({i,j}); if(it==ij.end()) continue; int k = it->second; G.d2[G.idx(i,j)] = v2[(size_t)k];
    }
    return true;
}

static double bilinear(const CrustGrid &G, const std::vector<double> &A, double lonDeg, double latDeg) {
    double lon = std::fmod(lonDeg + 180.0, 360.0); if(lon < 0) lon += 360.0; lon -= 180.0;
    double lat = std::max(-90.0, std::min(90.0, latDeg));
    double u = (lon - G.lon0) / G.dlon, v = (lat - G.lat0) / G.dlat;
    int i = (int)std::floor(u); double a = u - i; int j = (int)std::floor(v); double b = v - j;
    auto wrapI = [&](int ii){ ii%=G.nlon; if(ii<0) ii+=G.nlon; return ii; };
    auto clampJ = [&](int jj){ return std::max(0, std::min(G.nlat-1, jj)); };
    int i0=wrapI(i), i1=wrapI(i+1), j0=clampJ(j), j1=clampJ(j+1);
    double f00=A[G.idx(i0,j0)], f10=A[G.idx(i1,j0)], f01=A[G.idx(i0,j1)], f11=A[G.idx(i1,j1)];
    return (1-a)*(1-b)*f00 + a*(1-b)*f10 + (1-a)*b*f01 + a*b*f11;
}

// ============================================================================
//        OCC concentric spheres (with selective physical groups)  *** MOD ***
// ============================================================================
static void createConcentricSphericalLayers(
    const std::vector<double> &radii,
    double meshSizeMin, double meshSizeMax,
    int elementOrder, int algorithm,
    std::vector<int> &premVolumeTagsOut,
    std::vector<int> &outerTwoShellsOut // *** MOD *** return last two shell volume tags
) {
    premVolumeTagsOut.clear();
    outerTwoShellsOut.clear(); // *** MOD ***
    const int numLayers = (int)radii.size();
    if (numLayers < 1) { std::cerr << "Error: need at least one radius.\n"; return; }

    std::vector<int> sphereTags;
    for (int i = 0; i < numLayers; ++i) {
        int tag = gmsh::model::occ::addSphere(0, 0, 0, radii[i]);
        sphereTags.push_back(tag);
    }
    gmsh::model::occ::synchronize();

    int layerTag = 1; 
    int surfaceTag = 1;

    // innermost solid
    gmsh::model::addPhysicalGroup(3, {sphereTags[0]}, layerTag);
    gmsh::model::setPhysicalName(3, layerTag, "layer_1");
    premVolumeTagsOut.push_back(sphereTags[0]);

    // label its surface as surface_1
    {
        std::vector<std::pair<int, int>> surfaceEntities0;
        gmsh::model::getBoundary({{3, sphereTags[0]}}, surfaceEntities0, false, false, false);
        if(!surfaceEntities0.empty()) {
            std::pair<int, int> surface0 = surfaceEntities0[0];
            gmsh::model::addPhysicalGroup(2, {surface0.second}, surfaceTag);
            gmsh::model::setPhysicalName(2, surfaceTag, "surface_1");
        }
    }

    for (int i = 1; i < numLayers; ++i) {
        std::vector<std::pair<int, int>> ov;
        std::vector<std::vector<std::pair<int, int>>> ovv;
        gmsh::model::occ::cut({{3, sphereTags[i]}}, {{3, sphereTags[i-1]}}, ov, ovv, -1, false, false);
        gmsh::model::occ::synchronize();

        std::vector<int> vols;
        for (auto &p : ov) if (p.first == 3) vols.push_back(p.second);

        for (int v : vols) premVolumeTagsOut.push_back(v);

        bool isLastTwo = (i >= numLayers - 2); // *** MOD ***
        if(!isLastTwo) {
            ++layerTag;
            gmsh::model::addPhysicalGroup(3, vols, layerTag);
            gmsh::model::setPhysicalName(3, layerTag, "layer_" + std::to_string(i+1));

            for (const auto &vol : vols) {
                std::vector<std::pair<int, int>> surfaceEntities;
                gmsh::model::getBoundary({{3, vol}}, surfaceEntities, false, false, false);
                if(!surfaceEntities.empty()) {
                    std::pair<int, int> surface = surfaceEntities[0];
                    if (surface.first == 2) {
                        ++surfaceTag;
                        gmsh::model::addPhysicalGroup(2, {surface.second}, surfaceTag);
                        gmsh::model::setPhysicalName(2, surfaceTag, "surface_" + std::to_string(i+1));
                    }
                }
            }
        } else {
            for (int v : vols) outerTwoShellsOut.push_back(v); // *** MOD ***
        }
    }

    // Keep all geometric spheres; do not remove. (requested)

    // store options (meshing later)
    gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
    gmsh::option::setNumber("Mesh.Algorithm3D", algorithm);
    gmsh::option::setNumber("Mesh.MeshSizeMin", meshSizeMin);
    gmsh::option::setNumber("Mesh.MeshSizeMax", meshSizeMax);
}

// ============================================================================
//     Helpers to build OCC Moho/Top surfaces for Gmsh 4.14   *** MOD ***
// ============================================================================

// --- pick a far-away range of surface tags to avoid collisions ---  *** MOD ***
static int reserveSurfaceTagBaseOnce() {
    std::vector<std::pair<int,int>> faces;
    gmsh::model::getEntities(faces, 2);
    int mx = 0;
    for (auto &f : faces) mx = std::max(mx, f.second);
    return mx + 100000;
}
static int nextFarSurfaceTag() {
    static int base = -1;
    if (base < 0) base = reserveSurfaceTagBaseOnce();
    return ++base;
}

// sample grids (avoid exact poles; open seam in longitude)     *** MOD ***
static void sampleMohoTopGridOCC(const CrustGrid &G, double R_meter,
                                 int nLon, int nLat,
                                 std::vector<double> &Xmo, std::vector<double> &Ymo, std::vector<double> &Zmo,
                                 std::vector<double> &Xtp, std::vector<double> &Ytp, std::vector<double> &Ztp,
                                 double &rMoMin, double &rMoMax, double &rTpMin, double &rTpMax)
{
    Xmo.resize(nLon * nLat); Ymo.resize(nLon * nLat); Zmo.resize(nLon * nLat);
    Xtp.resize(nLon * nLat); Ytp.resize(nLon * nLat); Ztp.resize(nLon * nLat);

    rMoMin = 1e9; rMoMax = -1e9;
    rTpMin = 1e9; rTpMax = -1e9;

    for (int j = 0; j < nLat; ++j) {
        // avoid exact poles: center samples within (-90, +90)
        double lat = -90.0 + 180.0 * ((j + 0.5) / (double)nLat);  // *** MOD ***
        double phi = lat * M_PI / 180.0;
        double cphi = std::cos(phi), sphi = std::sin(phi);

        for (int i = 0; i < nLon; ++i) {
            // open seam: no duplication of last column            // *** MOD ***
            double lon = -180.0 + 360.0 * ((i + 0.5) / (double)nLon);
            double lam = lon * M_PI / 180.0;
            double clam = std::cos(lam), slam = std::sin(lam);

            double d1 = bilinear(G, G.d1, lon, lat);
            double d2 = bilinear(G, G.d2, lon, lat);
            double rMo = 1.0 + d2 / R_meter;
            double rTp = 1.0 + (d1 + d2) / R_meter;

            double x_u = cphi * clam, y_u = cphi * slam, z_u = sphi;

            int k = j * nLon + i;
            Xmo[k] = rMo * x_u; Ymo[k] = rMo * y_u; Zmo[k] = rMo * z_u;
            Xtp[k] = rTp * x_u; Ytp[k] = rTp * y_u; Ztp[k] = rTp * z_u;

            rMoMin = std::min(rMoMin, rMo); rMoMax = std::max(rMoMax, rMo);
            rTpMin = std::min(rTpMin, rTp); rTpMax = std::max(rTpMax, rTp);
        }
    }
    // no seam duplication; OCC B-spline is open in U
}

// Fallback: skin through latitude wires if B-spline fails
static int addSkinnedSurfaceFromGrid_ThruSections(const std::vector<double>& X,
                                                  const std::vector<double>& Y,
                                                  const std::vector<double>& Z,
                                                  int nU, int nV)
{
    // Build points as a grid
    std::vector<std::vector<int>> pt(nV, std::vector<int>(nU));
    for (int j = 0; j < nV; ++j)
        for (int i = 0; i < nU; ++i) {
            int k = j*nU + i;
            int p = gmsh::model::occ::addPoint(X[k], Y[k], Z[k]);
            pt[j][i] = p;
        }

    // Make a spline wire for each latitude row
    std::vector<int> wires; wires.reserve(nV);
    for (int j = 0; j < nV; ++j) {
        int crv = gmsh::model::occ::addSpline(pt[j]);
        int wr  = gmsh::model::occ::addWire({crv});
        wires.push_back(wr);
    }
    gmsh::model::occ::synchronize();

    // Gmsh 4.14 signature:
    // addThruSections(const std::vector<int>& wireTags,
    //                 gmsh::vectorpair& outDimTags,
    //                 int tag = -1, bool makeSolid = true, bool ruled = false,
    //                 int maxDegree = -1, const std::string& continuity = "",
    //                 const std::string& param = "", bool preserveParam = false)
    gmsh::vectorpair out;
    gmsh::model::occ::addThruSections(
        wires, out,
        /*tag*/        -1,
        /*makeSolid*/  false,
        /*ruled*/      false,
        /*maxDegree*/  -1,
        /*continuity*/ "",
        /*param*/      "",
        /*preserve*/   false
    );
    gmsh::model::occ::synchronize();

    // Pick a surface dimtag from 'out' (some builds can return more than one)
    // Choose the face with the largest bounding box diagonal.
    int faceTag = -1;
    double bestD = -1.0;
    for (const auto &dt : out) {
        if (dt.first != 2) continue;
        double xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(2, dt.second, xmin, ymin, zmin, xmax, ymax, zmax);
        double dx = xmax - xmin, dy = ymax - ymin, dz = zmax - zmin;
        double diag = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (diag > bestD) { bestD = diag; faceTag = dt.second; }
    }
    if (faceTag < 0)
        throw std::runtime_error("addThruSections: no face returned");

    return faceTag;
}

// Gmsh 4.14 API: addBSplineSurface(pointTags, nU, nV, tag, degreeU, weights, knotsU, knotsV, multU, multV, wireTags, wire3D)
static int addBSplineSurfaceFromGrid(const std::vector<double> &X,
                                     const std::vector<double> &Y,
                                     const std::vector<double> &Z,
                                     int nU, int nV,
                                     int degU = 3, int /*degV*/ = 3)
{
    std::vector<int> pTags;
    pTags.reserve(nU * nV);
    for (int j = 0; j < nV; ++j)
        for (int i = 0; i < nU; ++i) {
            int k = j * nU + i;
            int pt = gmsh::model::occ::addPoint(X[k], Y[k], Z[k]);
            pTags.push_back(pt);
        }

    static const std::vector<double> emptyD;
    static const std::vector<int>    emptyI;
    static const std::vector<int>    emptyWires;

    int explicitTag = nextFarSurfaceTag();
    gmsh::model::occ::synchronize();

    int tryDeg = std::min(degU, std::min(nU - 1, nV - 1));
    for (; tryDeg >= 1; --tryDeg) {
        try {
            int faceTag = gmsh::model::occ::addBSplineSurface(
                pTags, nU, nV,
                /*tag*/        explicitTag,
                /*degreeU*/    tryDeg,
                /*weights*/    emptyD,
                /*knotsU */    emptyD,
                /*knotsV */    emptyD,
                /*multU  */    emptyI,
                /*multV  */    emptyI,
                /*wireTags*/   emptyWires,
                /*wire3D */    false
            );
            return faceTag;
        } catch(const std::exception &) {
            // lower degree and retry
        }
    }
    // final fallback
    return addSkinnedSurfaceFromGrid_ThruSections(X, Y, Z, nU, nV);
}

// ============================================================================
//                 Utilities to detect faces in a volume  *** MOD ***
// ============================================================================
static bool volumeHasFace(int volTag, int faceTag) {
    std::vector<std::pair<int,int>> b;
    gmsh::model::getBoundary({{3, volTag}}, b, true, true, true);
    for (auto &e : b) if (e.first == 2 && e.second == faceTag) return true;
    return false;
}

static int findOutermostSurfaceByBBox() {
    std::vector<std::pair<int,int>> faces;
    gmsh::model::getEntities(faces, 2);
    auto bboxDiag = [](int tag){
        double xmin, ymin, zmin, xmax, ymax, zmax;
        gmsh::model::getBoundingBox(2, tag, xmin, ymin, zmin, xmax, ymax, zmax);
        double dx = xmax - xmin, dy = ymax - ymin, dz = zmax - zmin;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };
    int bestTag = -1; double bestD = -1.0;
    for (auto &f : faces) {
        double d = bboxDiag(f.second);
        if (d > bestD) { bestD = d; bestTag = f.second; }
    }
    return bestTag;
}

// ============================================================================
//       Build Moho/Top OCC surfaces and fragment shells   *** MOD ***
// ============================================================================
static void buildCrustShellFromCRUST_andFragment(
    const CrustGrid &G, double R_meter, int N,
    const std::vector<int> &outerTwoShells,
    std::vector<int> &crustVols_out,
    std::vector<int> &outerShellVols_out,
    int &surfMoho_out, int &surfTop_out, int &surfOuter_out
) {
    crustVols_out.clear();
    outerShellVols_out.clear();
    surfMoho_out = surfTop_out = surfOuter_out = -1;

    // 1) sample grids and create OCC faces
    const int nLon = 180;  // open seam in U (no duplication)        // *** MOD ***
    const int nLat =  91;  // (0.5)-shift avoids exact poles         // *** MOD ***

    std::vector<double> Xmo,Ymo,Zmo, Xtp,Ytp,Ztp;
    double rMoMin=0, rMoMax=0, rTpMin=0, rTpMax=0;
    sampleMohoTopGridOCC(G, R_meter, nLon, nLat, Xmo,Ymo,Zmo, Xtp,Ytp,Ztp, rMoMin,rMoMax, rTpMin,rTpMax);

    int faceMoho = addBSplineSurfaceFromGrid(Xmo, Ymo, Zmo, nLon, nLat, 3, 3);
    int faceTop  = addBSplineSurfaceFromGrid(Xtp, Ytp, Ztp, nLon, nLat, 3, 3);
    if (faceMoho <= 0 || faceTop <= 0) throw std::runtime_error("Failed to create Moho/Top OCC surfaces");
    gmsh::model::occ::synchronize();

    // 2) Label Moho=N-1, Top=N (surface physical IDs)
    {
        int idMoho = N - 1;
        int idTop  = N;
        gmsh::model::addPhysicalGroup(2, {faceMoho}, idMoho);
        gmsh::model::setPhysicalName(2, idMoho, "surface_moho");
        gmsh::model::addPhysicalGroup(2, {faceTop}, idTop);
        gmsh::model::setPhysicalName(2, idTop, "surface_top");
    }

    // 3) Label outermost surface as N+1
    gmsh::model::mesh::generate(2); // ensure faces exist for bbox
    int outerFace = findOutermostSurfaceByBBox();
    if (outerFace > 0) {
        int idOuter = N + 1;
        gmsh::model::addPhysicalGroup(2, {outerFace}, idOuter);
        gmsh::model::setPhysicalName(2, idOuter, "surface_outer");
    }
    surfMoho_out = faceMoho;
    surfTop_out  = faceTop;
    surfOuter_out = outerFace;

    // 4) Fragment the two outermost spherical shells with Moho & Top faces
    if(outerTwoShells.size() < 2) {
        std::cerr << "Warning: expected 2 outer shell volumes; got " << outerTwoShells.size() << "\n";
    }
    std::vector<std::pair<int,int>> toFrag;
    for (int v : outerTwoShells) toFrag.push_back({3, v});

    std::vector<std::pair<int,int>> tools = { {2, faceMoho}, {2, faceTop} };

    std::vector<std::pair<int,int>> out;
    std::vector<std::vector<std::pair<int,int>>> outmap;
    if(!toFrag.empty()) {
        gmsh::model::occ::fragment(toFrag, tools, out, outmap);
        gmsh::model::occ::synchronize();
    }

    // 5) Classify resulting fragment volumes into crust and outer_shell
    std::vector<std::pair<int,int>> vols;
    gmsh::model::getEntities(vols, 3);

    for (auto &v : vols) {
        int vt = v.second;
        bool hasMoho = (faceMoho > 0) ? volumeHasFace(vt, faceMoho) : false;
        bool hasTop  = (faceTop  > 0) ? volumeHasFace(vt, faceTop ) : false;
        bool hasOuter = (outerFace > 0) ? volumeHasFace(vt, outerFace) : false;

        if (hasMoho && hasTop) {
            crustVols_out.push_back(vt);
        } else if (!hasMoho && hasTop && hasOuter) {
            outerShellVols_out.push_back(vt);
        }
    }

    if(!crustVols_out.empty()) {
        int pgCrust = gmsh::model::addPhysicalGroup(3, crustVols_out);
        gmsh::model::setPhysicalName(3, pgCrust, "crust");
    }
    if(!outerShellVols_out.empty()) {
        int pgOuter = gmsh::model::addPhysicalGroup(3, outerShellVols_out);
        gmsh::model::setPhysicalName(3, pgOuter, "outer_shell");
    }
}

// ============================================================================
//                                 CLI parse
// ============================================================================
std::vector<double> parseDoubles(const std::string &radiiStr) {
    std::vector<double> radii;
    std::istringstream iss(radiiStr);
    std::string token;

    while (std::getline(iss, token, '-')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        radii.push_back(std::stod(token));
    }

    return radii;
}

// ============================================================================
//                                   main
// ============================================================================
int main(int argc, char **argv) {
    double meshSizeMin = 30e3;
    double meshSizeMax = 300e3;
    int algorithm = 1;
    int elementOrder = 1;
    std::string inputFileName = "data/prem.nocrust";
    std::string outputFileName = "mesh/prem_with_crust";
    std::string crustFile_d1 = "data/crust-1.0/crsthk.xyz";      // (lon lat d1, positive)
    std::string crustFile_d2 = "data/crust-1.0/depthtomoho.xyz"; // (lon lat d2, negative)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            inputFileName = argv[++i];
        } else if (arg == "-s" && i + 1 < argc) {
            auto meshSizes = parseDoubles(argv[++i]);
            if (meshSizes.size() == 2) {
                meshSizeMin = meshSizes[0];
                meshSizeMax = meshSizes[1];
            } else {
                std::cerr << "Error: mesh sizes should have two values.\n";
                return 1;
            }
        } else if (arg == "-o" && i + 1 < argc) {
            outputFileName = argv[++i];
        } else if (arg == "-order" && i + 1 < argc) {
            elementOrder = std::stoi(argv[++i]);
        } else if (arg == "-alg" && i + 1 < argc) {
            algorithm = std::stoi(argv[++i]);
        }
    }

    gmsh::initialize();
    gmsh::option::setNumber("General.Verbosity", 2);      // optional, quieter logs = 1
    gmsh::option::setNumber("Geometry.OCCFixDegenerated", 1);
    gmsh::option::setNumber("Geometry.OCCFixSmallEdges", 1);
    gmsh::option::setNumber("Geometry.OCCSewFaces", 1);
    gmsh::model::add("Earth_PREM_CRUST");

    double R;
    std::vector<double> radii = extractLayerBoundaries(inputFileName, R);
    meshSizeMin /= R;
    meshSizeMax /= R;

    std::cout << "Detected radii of " << radii.size() << " layers: ";
    for (const double r : radii) {
        std::cout << std::fixed << std::setprecision(8) << r << " ";
    }
    std::cout << std::fixed << std::setprecision(2) << "(The length scale is " << R << " meters.)" << std::endl;

    const int N = (int)radii.size(); // *** MOD ***

    std::vector<int> premVols;
    std::vector<int> outerTwoShells; // *** MOD ***
    createConcentricSphericalLayers(radii, meshSizeMin, meshSizeMax, elementOrder, algorithm, premVols, outerTwoShells);

    CrustGrid G;
    if(!buildCrustGrid(crustFile_d1, crustFile_d2, G)) {
        std::cerr << "Warning: could not build CRUST grid; proceeding without variable crust.\n";
    } else {
        std::vector<int> crustVols, outerShellVols;
        int surfMoho=-1, surfTop=-1, surfOuter=-1;

        buildCrustShellFromCRUST_andFragment(G, R, N, outerTwoShells,
                                             crustVols, outerShellVols,
                                             surfMoho, surfTop, surfOuter);
    }

    gmsh::option::setNumber("Mesh.Algorithm3D", algorithm);
    gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
    gmsh::option::setNumber("Mesh.Optimize", 1);
    gmsh::option::setNumber("Mesh.OptimizeNetgen", 1);
    gmsh::model::mesh::generate(3);
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write(outputFileName + ".msh");
    std::cout << "Wrote " << outputFileName << ".msh\n";
    gmsh::finalize();

    return 0;
}

