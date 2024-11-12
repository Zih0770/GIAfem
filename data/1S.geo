Mesh.MshFileVersion = 2.2;

SetFactory("OpenCASCADE");

lc = 1;

Sphere(1) = {0.0, 0.0, 0.0, 8.0}; 

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

//Field[1] = Distance;
//Field[1].PointsList = {1};
//Field[1].Sampling = 100;

//Field[2] = Threshold;
//Field[2].InField = 1;
//Field[2].SizeMin = lc;
//Field[2].SizeMax = 100*lc;
//Field[2].DistMin = 6397.0;
//Field[2].DistMax = 100000.0;

Physical Volume(301) = {1};
//Physical Volume(306) = {5};
//Physical Volume(307) = {6};

Physical Surface(201) = {1};


//Mesh.Algorithm = 5;
