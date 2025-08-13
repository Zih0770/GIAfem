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
#include <set>
#include <functional>
#include <limits>

// ======================== CLI helpers ========================
static std::vector<double> parseDoubles(const std::string &s) {
    std::vector<double> v; std::istringstream iss(s); std::string tok;
    while (std::getline(iss, tok, '-')) {
        tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
        if (!tok.empty()) v.push_back(std::stod(tok));
    }
    return v;
}

static std::vector<double>
extractLayerBoundaries(const std::string &fileName, double Rref, double &R_out)
{
    R_out = Rref;

    std::ifstream file(fileName);
    if (!file) { std::cerr << "Error: Unable to open " << fileName << "\n"; return {}; }

    std::string line;
    int lineCount = 0;

    std::vector<double> allR_m;
    std::vector<double> ifaces_m;

    double prevR = std::numeric_limits<double>::quiet_NaN();
    while (std::getline(file, line)) {
        if (lineCount < 3) { lineCount++; continue; }
        std::istringstream iss(line);
        double r, density, pWave, sWave, bulkM, shearM;
        if (!(iss >> r >> density >> pWave >> sWave >> bulkM >> shearM)) continue;

        if (!std::isnan(prevR) && std::abs(r - prevR) < 1e-6) {
            ifaces_m.push_back(r);
        }
        prevR = r;
        allR_m.push_back(r);
    }

    if (allR_m.empty()) {
        std::cerr << "Error: no radii read from " << fileName << "\n";
        return {};
    }

    const double rMoho_m = allR_m.back();
    std::cout << std::fixed << std::setprecision(2)
              << "Info: last PREM radius (treated as Moho) = " << rMoho_m
              << " m; using sea-level Rref = " << Rref << " m.\n";

    std::vector<double> radii;
    radii.reserve(ifaces_m.size() + 2);
    for (double rm : ifaces_m) radii.push_back(rm / Rref);
    radii.push_back(1.0);
    radii.push_back(1.2);
    return radii;
}

// ======================== CRUST-1.0 grid =====================
struct CrustGrid {
    int nlon = 0, nlat = 0;
    double lon0 = -180.0, lat0 = -90.0;
    double dlon = 1.0, dlat = 1.0;
    std::vector<double> d1_km; // crustal thickness (km, +)
    std::vector<double> d2_km; // depth to Moho (km, -)
    int idx(int i, int j) const { return j * nlon + i; }
};

static bool loadXYZ(const std::string &fname, std::vector<double> &lon, std::vector<double> &lat, std::vector<double> &val) {
    std::ifstream f(fname);
    if (!f) return false;
    lon.clear(); lat.clear(); val.clear();
    double a,b,c;
    while (f >> a >> b >> c) { lon.push_back(a); lat.push_back(b); val.push_back(c); }
    return !lon.empty();
}

static bool buildCrustGrid(const std::string &f1, const std::string &f2, CrustGrid &G) {
    std::vector<double> lon1, lat1, v1, lon2, lat2, v2;
    if (!loadXYZ(f1, lon1, lat1, v1)) { std::cerr << "Error: cannot read " << f1 << "\n"; return false; }
    if (!loadXYZ(f2, lon2, lat2, v2)) { std::cerr << "Error: cannot read " << f2 << "\n"; return false; }
    if (lon1.size() != lon2.size()) { std::cerr << "Error: crust xyz sizes differ\n"; return false; }

    std::vector<double> ulon = lon1, ulat = lat1;
    std::sort(ulon.begin(), ulon.end()); ulon.erase(std::unique(ulon.begin(), ulon.end()), ulon.end());
    std::sort(ulat.begin(), ulat.end()); ulat.erase(std::unique(ulat.begin(), ulat.end()), ulat.end());
    G.nlon = (int)ulon.size(); G.nlat = (int)ulat.size();
    G.lon0 = ulon.front(); G.lat0 = ulat.front();
    G.dlon = (ulon.back() - ulon.front()) / (G.nlon - 1);
    G.dlat = (ulat.back() - ulat.front()) / (G.nlat - 1);
    G.d1_km.assign(G.nlon * G.nlat, 0.0);
    G.d2_km.assign(G.nlon * G.nlat, 0.0);

    auto normLon = [](double L){ double x = std::fmod(L + 180.0, 360.0); if(x < 0) x += 360.0; return x - 180.0; };

    std::map<std::pair<int,int>, int> ij;
    for (size_t k=0; k<lon1.size(); ++k) {
        double L = normLon(lon1[k]);
        double B = std::max(-90.0, std::min(90.0, lat1[k]));
        int i = (int)std::llround((L - G.lon0) / G.dlon);
        int j = (int)std::llround((B - G.lat0) / G.dlat);
        if(i>=0 && i<G.nlon && j>=0 && j<G.nlat) ij[{i,j}] = (int)k;
    }
    for (int j=0; j<G.nlat; ++j) for (int i=0; i<G.nlon; ++i) {
        auto it = ij.find({i,j}); if(it==ij.end()) continue; int k = it->second; G.d1_km[G.idx(i,j)] = v1[k]; //
    }
    ij.clear();
    for (size_t k=0; k<lon2.size(); ++k) {
        double L = normLon(lon2[k]);
        double B = std::max(-90.0, std::min(90.0, lat2[k]));
        int i = (int)std::llround((L - G.lon0) / G.dlon);
        int j = (int)std::llround((B - G.lat0) / G.dlat);
        if(i>=0 && i<G.nlon && j>=0 && j<G.nlat) ij[{i,j}] = (int)k;
    }
    for (int j=0; j<G.nlat; ++j) for (int i=0; i<G.nlon; ++i) {
        auto it = ij.find({i,j}); if(it==ij.end()) continue; int k = it->second; G.d2_km[G.idx(i,j)] = v2[k];
    }
    return true;
}

static double bilinear(const CrustGrid &G, const std::vector<double> &A, double lonDeg, double latDeg) {
    double lon = std::fmod(lonDeg + 180.0, 360.0); if (lon < 0) lon += 360.0; lon -= 180.0;
    double lat = std::max(-90.0, std::min(90.0, latDeg));
    double u = (lon - G.lon0) / G.dlon, v = (lat - G.lat0) / G.dlat;
    int i = (int)std::floor(u); double a = u - i;
    int j = (int)std::floor(v); double b = v - j;
    auto wrapI = [&](int ii){ ii%=G.nlon; if(ii<0) ii+=G.nlon; return ii; };
    auto clampJ = [&](int jj){ return std::max(0, std::min(G.nlat-1, jj)); };
    int i0=wrapI(i), i1=wrapI(i+1), j0=clampJ(j), j1=clampJ(j+1);
    double f00=A[G.idx(i0,j0)], f10=A[G.idx(i1,j0)], f01=A[G.idx(i0,j1)], f11=A[G.idx(i1,j1)];
    return (1-a)*(1-b)*f00 + a*(1-b)*f10 + (1-a)*b*f01 + a*b*f11;
}

struct DiscreteTriMesh {
    int surfTag = -1;
    std::vector<std::size_t> nodeTags;
    std::vector<double> coords;
    std::vector<std::size_t> triTags;
    std::vector<std::size_t> triConn;
};

static DiscreteTriMesh buildIcosphereDiscrete(int subdivLevel) { //
    // initial icosahedron 
    struct Tri { int a,b,c; };
    const double phi = (1.0 + std::sqrt(5.0)) * 0.5;
    std::vector<std::array<double,3>> V = {
        {-1,  phi, 0}, { 1,  phi, 0}, {-1, -phi, 0}, { 1, -phi, 0},
        {0, -1,  phi}, {0,  1,  phi}, {0, -1, -phi}, {0,  1, -phi},
        { phi, 0, -1}, { phi, 0,  1}, {-phi, 0, -1}, {-phi, 0,  1}
    };
    auto norm = [](std::array<double,3> &p){
        double r = std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
        p[0]/=r; p[1]/=r; p[2]/=r;
    };
    for (auto &p: V) norm(p);

    std::vector<Tri> F = {
        {0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
        {1,5,9}, {5,11,4}, {11,10,2}, {10,7,6}, {7,1,8},
        {3,9,4}, {3,4,2}, {3,2,6}, {3,6,8}, {3,8,9},
        {4,9,5}, {2,4,11}, {6,2,10}, {8,6,7}, {9,8,1}
    };

    // recursive subdivision
    auto edgeKey = [](int i, int j){ return std::pair<int,int>(std::min(i,j), std::max(i,j)); };
    for (int s=0; s<subdivLevel; ++s) {
        std::map<std::pair<int,int>, int> midCache;
        auto midpoint = [&](int i, int j)->int {
            auto key = edgeKey(i,j);
            auto it = midCache.find(key);
            if (it != midCache.end()) return it->second;
            std::array<double,3> p = { (V[i][0]+V[j][0])*0.5, (V[i][1]+V[j][1])*0.5, (V[i][2]+V[j][2])*0.5 };
            norm(p);
            int idx = (int)V.size();
            V.push_back(p);
            midCache[key] = idx;
            return idx;
        };
        std::vector<Tri> F2;
        F2.reserve(F.size()*4);
        for (const auto &t : F) {
            int a=t.a, b=t.b, c=t.c;
            int ab=midpoint(a,b), bc=midpoint(b,c), ca=midpoint(c,a);
            F2.push_back({a, ab, ca});
            F2.push_back({b, bc, ab});
            F2.push_back({c, ca, bc});
            F2.push_back({ab, bc, ca});
        }
        F.swap(F2);
    }

    DiscreteTriMesh M;
    const int nn = (int)V.size();
    const int nt = (int)F.size();

    M.nodeTags.resize(nn);
    M.coords.resize(3 * nn);
    for (int i=0;i<nn;++i) {
        M.nodeTags[i] = i+1;
        M.coords[3*i+0] = V[i][0];
        M.coords[3*i+1] = V[i][1];
        M.coords[3*i+2] = V[i][2];
    }
    M.triTags.resize(nt);
    M.triConn.resize(3 * nt);
    for (int t=0; t<nt; ++t) {
        M.triTags[t] = t+1;
        M.triConn[3*t+0] = (std::size_t)F[t].a + 1; //
        M.triConn[3*t+1] = (std::size_t)F[t].b + 1;
        M.triConn[3*t+2] = (std::size_t)F[t].c + 1;
    }

    M.surfTag = gmsh::model::addDiscreteEntity(2, 100);
    gmsh::model::mesh::addNodes(2, M.surfTag, M.nodeTags, M.coords);
    std::vector<int> etypes = {2};
    std::vector<std::vector<std::size_t>> tagBlocks(1), connBlocks(1);
    tagBlocks[0] = M.triTags;
    connBlocks[0] = M.triConn;
    gmsh::model::mesh::addElements(2, M.surfTag, etypes, tagBlocks, connBlocks);
    return M;
}

static int makeDiscreteCopyWithRadii(const DiscreteTriMesh &seed,
                                     const std::vector<double> &radUnit,
                                     int surfTag,
                                     std::size_t nodeBase,
                                     std::size_t elemBase)
{
    if (radUnit.size() * 3 != seed.coords.size())
        throw std::runtime_error("radUnit size mismatch");

    const int nn = (int)seed.nodeTags.size();
    const int nt = (int)(seed.triConn.size()/3);

    std::vector<std::size_t> nodeTags(nn);
    std::vector<double> xyz(3 * nn);
    for (int a=0; a<nn; ++a) {
        nodeTags[a] = nodeBase + (std::size_t)a + 1;
        double x = seed.coords[3*a], y = seed.coords[3*a+1], z = seed.coords[3*a+2];
        double r = radUnit[a];
        xyz[3*a+0] = r * x;
        xyz[3*a+1] = r * y;
        xyz[3*a+2] = r * z;
    }

    std::vector<std::size_t> triTags(nt);
    std::vector<std::size_t> triConn(3 * nt);
    for (int t = 0; t < nt; ++t) {
        triTags[t] = elemBase + (std::size_t)t + 1;
        for (int k = 0; k < 3; ++k) {
            std::size_t oldNode1 = seed.triConn[3*t + k];
            std::size_t a = oldNode1 - 1;
            triConn[3*t + k] = nodeTags[a];
        }
    }

    int s = gmsh::model::addDiscreteEntity(2, surfTag);
    gmsh::model::mesh::addNodes(2, s, nodeTags, xyz);
    std::vector<int> etypes = {2};
    std::vector<std::vector<std::size_t>> tagBlocks(1), connBlocks(1);
    tagBlocks[0] = triTags;
    connBlocks[0] = triConn;
    gmsh::model::mesh::addElements(2, s, etypes, tagBlocks, connBlocks);
    return s;
}

static std::vector<double> constantRadiusField(const DiscreteTriMesh &seed, double rconst) {
    return std::vector<double>(seed.nodeTags.size(), rconst);
}

static void computeMohoTopRadii(const DiscreteTriMesh &seed,
                                const CrustGrid &G, double Rref,
                                std::vector<double> &rMoho, std::vector<double> &rTop,
                                double &rMohoRep, double &rTopRep)
{
    rMoho.resize(seed.nodeTags.size());
    rTop.resize(seed.nodeTags.size());

    auto nodeLonLat = [&](int a)->std::pair<double,double> {
        double x = seed.coords[3*a], y = seed.coords[3*a+1], z = seed.coords[3*a+2];
        double r = std::sqrt(x*x+y*y+z*z);
        double lon = std::atan2(y,x) * 180.0/M_PI;
        double lat = std::asin(z/r) * 180.0/M_PI;
        return {lon, lat};
    };

    double sumMoho = 0.0, sumTop = 0.0;
    const int nn = (int)seed.nodeTags.size();
    const double km2R = 1000.0 / Rref;
    const double eps  = 1e-6;

    for (int a=0; a<nn; ++a) {
        auto [lon, lat] = nodeLonLat(a);
        double d1_km = bilinear(G, G.d1_km, lon, lat);
        double d2_km = bilinear(G, G.d2_km, lon, lat);
        double d1 = d1_km * km2R;
        double d2 = d2_km * km2R;

        double rM = 1.0 + d2;            // Moho radius
        double rT = 1.0 + (d1 + d2);     // Topography radius
        //if (rT < rM + eps) rT = rM + eps;

        rMoho[a] = rM;
        rTop[a]  = rT;
        sumMoho += rM; sumTop += rT;
    }
    rMohoRep = sumMoho / nn;
    rTopRep  = sumTop  / nn;
}

//Utilities for bucketing
static double meanRadiusFromNodes(int surfTag) {
    std::vector<std::size_t> nodeTags;
    std::vector<double> coords, dummy;
    gmsh::model::mesh::getNodes(nodeTags, coords, dummy, 2, surfTag, false, false);
    if (coords.empty()) return 0.0;
    const std::size_t nn = coords.size() / 3;
    double sum = 0.0;
    for (std::size_t i = 0; i < nn; ++i) {
        double x = coords[3*i+0], y = coords[3*i+1], z = coords[3*i+2];
        sum += std::sqrt(x*x + y*y + z*z);
    }
    return sum / (double)nn;
}

static std::vector<std::vector<int>>
binGeoSurfacesByRadius(const std::vector<int> &geoSurfs,
                              const std::vector<double> &targets_in)
{
    const int M = (int)targets_in.size();
    if (M == 0) return {};
    if (M == 1) {             
        return {geoSurfs};
    }

    std::vector<double> B(M - 1);
    for (int i = 0; i + 1 < M; ++i)
        B[i] = 0.5 * (targets_in[i] + targets_in[i + 1]);

    // Assign each surface to nearest target radius
    std::vector<std::vector<int>> buckets(M);
    for (int sTag : geoSurfs) {
        double r = meanRadiusFromNodes(sTag);          
        int k = (int)(std::upper_bound(B.begin(), B.end(), r) - B.begin());
        buckets[k].push_back(sTag);
    }

    // Debug: how many patches per target
    for (int i = 0; i < M; ++i) {
        std::cerr << "Target[" << i << "]=" << std::setprecision(8) << targets_in[i]
                  << " gets " << buckets[i].size() << " patches\n";
        if (buckets[i].empty())
            std::cerr << "  [warn] no patches near target[" << i << "]\n";
    }
    return buckets;
}

static std::vector<int> collectAllParamSurfaces() {
    std::vector<std::pair<int,int>> ents;
    gmsh::model::getEntities(ents, 2);
    std::vector<int> out;
    for (auto &p : ents) if (p.first == 2) out.push_back(p.second);
    return out;
}

int main(int argc, char **argv) {
    double meshSizeMin_m = 30e3;
    double meshSizeMax_m = 300e3;
    int elementOrder = 1;
    int alg3d = 10;              // HXT
    int alg2d = 6;              // Frontal-Delaunay
    double angle_deg = 180.0;    // classify angle
    double Rref = 6371000.0;     // sea-level reference radius

    std::string inputFileName = "data/prem.nocrust";
    std::string outputFileName = "mesh/prem_with_crust";
    std::string crustFile_d1 = "data/crust-1.0/crsthk.xyz";
    std::string crustFile_d2 = "data/crust-1.0/depthtomoho.xyz";

    for (int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i+1<argc) inputFileName = argv[++i];
        else if (arg == "-o" && i+1<argc) outputFileName = argv[++i];
        else if (arg == "-s" && i+1<argc) {
            auto ms = parseDoubles(argv[++i]);
            if (ms.size()==2) { meshSizeMin_m = ms[0]; meshSizeMax_m = ms[1]; }
            else { std::cerr << "Error: -s needs a-b\n"; return 1; }
        } else if (arg == "-order" && i+1<argc) elementOrder = std::stoi(argv[++i]);
        else if (arg == "-alg" && i+1<argc) alg3d = std::stoi(argv[++i]);
        else if (arg == "-alg2d" && i+1<argc) alg2d = std::stoi(argv[++i]);
        else if (arg == "-angle_deg" && i+1<argc) angle_deg = std::stod(argv[++i]);
        else if (arg == "-Rref" && i+1<argc) Rref = std::stod(argv[++i]);
        else if (arg == "-crust_d1" && i+1<argc) crustFile_d1 = argv[++i];
        else if (arg == "-crust_d2" && i+1<argc) crustFile_d2 = argv[++i];
    }

    gmsh::initialize();
    gmsh::model::add("Earth_PREM_CRUST_GEO");

    double Rmeter;
    auto radii = extractLayerBoundaries(inputFileName, Rref, Rmeter);
    if (radii.size() < 3) { std::cerr << "Not enough PREM radii\n"; gmsh::finalize(); return 1; }

    const int N = (int)radii.size(); // includes 1.0 at N-2 and 1.2 at N-1
    std::cout << "Detected radii of " << N << " layers (without Moho+topo+outer): ";
    for (double r : radii) std::cout << std::fixed << std::setprecision(8) << r << " ";
    std::cout << "(The length scale is " << std::fixed << std::setprecision(2) << Rmeter << " meters.)\n";

    // Seed sphere
    int Ns = 4;
    DiscreteTriMesh seed = buildIcosphereDiscrete(Ns);

    std::vector<int> discSpheres; discSpheres.reserve(N);
    std::vector<double> targetRadiiOrdered;
    int tag2 = 1;

    std::size_t nodeBase = seed.nodeTags.size();
    std::size_t elemBase = seed.triTags.size();

    auto add_surface = [&](const std::vector<double>& rad, int tag){
        int s = makeDiscreteCopyWithRadii(seed, rad, tag, nodeBase, elemBase);
        nodeBase += seed.nodeTags.size(); //
        elemBase += seed.triTags.size();
        return s;
    };

    for (int i=0; i<=N-3; ++i) {
        auto rad = constantRadiusField(seed, radii[i]);
        int s = add_surface(rad, tag2); ++tag2;
        discSpheres.push_back(s);
        targetRadiiOrdered.push_back(radii[i]);
        std::cout<<"Produced surface "<<s<<" with radius "<<std::fixed<<std::setprecision(8)<<radii[i]<<std::endl;
    }

    // CRUST Moho & Top
    CrustGrid G;
    bool haveCrust = buildCrustGrid(crustFile_d1, crustFile_d2, G);
    int discMoho = -1, discTop = -1;
    double rMohoRep = 1.0, rTopRep = 1.0;

    if (haveCrust) {
        std::vector<double> rM, rT;
        computeMohoTopRadii(seed, G, Rmeter, rM, rT, rMohoRep, rTopRep);
        discMoho = add_surface(rM, tag2); ++tag2;
        discTop  = add_surface(rT, tag2); ++tag2;
        targetRadiiOrdered.push_back(rMohoRep);
        std::cout<<"Produced Moho surface "<<discMoho<<" avg r="<<std::fixed<<std::setprecision(8)<<rMohoRep<<std::endl;
        targetRadiiOrdered.push_back(rTopRep);
        std::cout<<"Produced topography surface "<<discTop<<" avg r="<<std::fixed<<std::setprecision(8)<<rTopRep<<std::endl;
    } else {
        auto radSea = constantRadiusField(seed, radii[N-2]);
        int s = add_surface(radSea, tag2); ++tag2;
        discSpheres.push_back(s);
        targetRadiiOrdered.push_back(radii[N-2]);
        std::cout<<"Produced surface "<<s<<" with radius "<<std::fixed<<std::setprecision(8)<<radii[N-2]<<std::endl;
    }

    // Outermost sphere
    {
        auto radOuter = constantRadiusField(seed, radii[N-1]);
        int s = add_surface(radOuter, tag2); ++tag2;
        discSpheres.push_back(s);
        targetRadiiOrdered.push_back(radii[N-1]);
        std::cout<<"Produced the outermost surface "<<s<<" with radius "<<std::fixed<<std::setprecision(8)<<radii[N-1]<<std::endl;
    }

    gmsh::model::removeEntities({{2, seed.surfTag}}, true);
    std::cout<<"Surface "<<seed.surfTag<<" removed"<<std::endl;

    //Discrete -> GEO param surfaces
    const double angle = angle_deg * M_PI/180.0;
    const bool boundary = false;
    const bool forRep = true;
    const double curAngle = M_PI;
    const bool force = true;

    gmsh::model::mesh::classifySurfaces(angle, boundary, forRep, curAngle, force);
    gmsh::model::mesh::createTopology();
    gmsh::model::mesh::createGeometry();
    //gmsh::model::mesh::removeDuplicateNodes();
    gmsh::model::geo::synchronize();

    std::vector<int> allSurfs = collectAllParamSurfaces();
    if (allSurfs.empty()) { gmsh::finalize(); throw std::runtime_error("No parametrized surfaces"); }

    auto buckets = binGeoSurfacesByRadius(allSurfs, targetRadiiOrdered); //
    for (size_t k=0; k<buckets.size(); ++k) {
        if (buckets[k].empty()) {
            std::ostringstream oss;
            oss << "No patches assigned to target radius index " << k
                << " (target r=" << std::setprecision(8) << targetRadiiOrdered[k] << ")";
            gmsh::finalize();
            throw std::runtime_error(oss.str());
        }
    }

    const int L = (int)buckets.size();
    /*std::vector<int> layerSurf(L);
    for (int i = 0; i < L; ++i) {
        gmsh::model::mesh::setCompound(2, buckets[i]);
        layerSurf[i] = buckets[i].front();
    }
    gmsh::model::geo::synchronize();*/

    // SurfaceLoops & Volumes
    std::vector<int> loopTags(L);
    for (int i = 0; i < L; ++i)
        loopTags[i] = gmsh::model::geo::addSurfaceLoop(buckets[i]);
    gmsh::model::geo::synchronize();

    std::vector<int> volTags;
    volTags.reserve(L);
    volTags.push_back(gmsh::model::geo::addVolume({loopTags[0]}, 1));
    for (int i=1; i<L; ++i)
        volTags.push_back(gmsh::model::geo::addVolume({loopTags[i], loopTags[i-1]}, i+1));
    gmsh::model::geo::synchronize();

    // Physical groups
    for (int i = 0; i < L; ++i) {
        int physS = i + 1, physV = i + 1;
        gmsh::model::addPhysicalGroup(2, buckets[i], physS);
        gmsh::model::setPhysicalName(2, physS, "surface_" + std::to_string(physS));
        gmsh::model::addPhysicalGroup(3, {volTags[i]}, physV);
        gmsh::model::setPhysicalName(3, physV, "volume_" + std::to_string(physV));
    }

    const double hmin = meshSizeMin_m / Rmeter;
    const double hmax = meshSizeMax_m / Rmeter;

    double fac = 10;   
    const double distMin = 0.0;
    const double distMax = fac * hmin;

    std::vector<double> facesListD;
    for (const auto& b : buckets)
        for (int s : b) facesListD.push_back(s);

    int fDist = gmsh::model::mesh::field::add("Distance");
    gmsh::model::mesh::field::setNumbers(fDist, "FacesList", facesListD);
    //gmsh::model::mesh::field::setNumber(fDist, "Sampling", 100);

    int fTh = gmsh::model::mesh::field::add("Threshold");
    gmsh::model::mesh::field::setNumber(fTh, "InField", fDist);
    gmsh::model::mesh::field::setNumber(fTh, "SizeMin", hmin);
    gmsh::model::mesh::field::setNumber(fTh, "SizeMax", hmax);
    gmsh::model::mesh::field::setNumber(fTh, "DistMin", distMin);
    gmsh::model::mesh::field::setNumber(fTh, "DistMax", distMax);

    gmsh::model::mesh::field::setAsBackgroundMesh(fTh);

    //gmsh::option::setNumber("Mesh.Algorithm", alg2d);
    gmsh::option::setNumber("Mesh.MeshSizeMin", hmin);
    gmsh::option::setNumber("Mesh.MeshSizeMax", hmax);
    gmsh::option::setNumber("Mesh.ElementOrder", elementOrder);
    gmsh::option::setNumber("Mesh.Algorithm3D", alg3d);
    //gmsh::option::setNumber("Mesh.Optimize", 1);
    //gmsh::option::setNumber("Mesh.OptimizeNetgen", 1);

    // Generate
    //gmsh::model::mesh::generate(2);
    gmsh::model::mesh::generate(3);

    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::write(outputFileName + ".msh");
    std::cout << "Wrote " << outputFileName << ".msh\n";

    gmsh::finalize();
    return 0;
}

