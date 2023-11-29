l = 5e5;
lc = l/10;
Point(1) = {-l,-l,0,lc};
Point(2) = {l,-l,0,lc};
Point(3) = {l,l,0,lc};
Point(4) = {-l,l,0,lc};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

Physical Line("Wall") = {1,2,3,4};
Physical Surface("Domain") = {1};
Mesh.Algorithm=5;
