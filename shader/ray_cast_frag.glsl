#version 460 core

#define PI 3.1415926
#define TWO_PI 6.28318530718
#define EPSILON 0.0001f
#define MAX_OBJECTS 20

//////////////////////////////////////////////////////////////
//     Structures
//
struct BoxData {
  vec2 leftTop;
  vec2 rightBottom;
  vec3 color;
  bool isLight;
};

struct TriangleData {
  vec2 point1;
  vec2 point2;
  vec2 point3;
  vec3 color;
  bool isLight;
};

struct CircleData {
  vec2 center;
  vec2 radius;
  vec3 color;
  bool isLight;
};

struct Factor {
  vec4 angle;
  vec3 color;
  bool isLight;
};

struct Vec4Array {
  vec4 data[MAX_OBJECTS];
  float loss[MAX_OBJECTS];
  int length;
};

//////////////////////////////////////////////////////////////
//     Uniform buffers
//
layout(std430, binding = 0) buffer Boxes { BoxData boxes[]; };
layout(std430, binding = 1) buffer Triangles { TriangleData triangles[]; };
layout(std430, binding = 2) buffer Circles { CircleData circles[]; };

uniform int boxCount;
uniform int triangleCount;
uniform int circleCount;

in vec2 fragUV;
out vec4 fragColor;

//////////////////////////////////////////////////////////////
//     Optimized utility functions
//

/**
 * Computes the 2D cross product of two vectors
 * @param U First vector
 * @param V Second vector
 * @return Cross product value
 */
float cross2D(in vec2 U, in vec2 V) { return U.x * V.y - U.y * V.x; }

/**
 * Safe vector length calculation with minimum threshold
 * @param V Input vector
 * @return Length of vector (minimum 1e-9)
 */
float safeLength(vec2 V) { return max(length(V), 1e-9); }

/**
 * Checks if point W is in the minor arc between points U and V
 * @param U First arc point
 * @param V Second arc point
 * @param W Point to check
 * @return 1.0 if in arc, 0.0 otherwise
 */
float inMinorArc(in vec2 U, in vec2 V, in vec2 W) {
  vec2 normalizedU = U / safeLength(U);
  vec2 normalizedV = V / safeLength(V);
  vec2 normalizedW = W / safeLength(W);

  float isCounterClockwise = step(0.0, cross2D(normalizedU, normalizedV));
  float condition1 = step(0.0, cross2D(normalizedU, normalizedW)) *
  step(0.0, cross2D(normalizedW, normalizedV));
  float condition2 = step(0.0, cross2D(normalizedV, normalizedW)) *
  step(0.0, cross2D(normalizedW, normalizedU));

  return isCounterClockwise * condition1 +
  (1.0 - isCounterClockwise) * condition2;
}

/**
 * Calculates the angle between two vectors
 * @param P First vector
 * @param Q Second vector
 * @return Angle in radians
 */
float angleBetween(in vec2 P, in vec2 Q) {
  float lengthP = safeLength(P);
  float lengthQ = safeLength(Q);
  float denominator = max(lengthP * lengthQ, 1e-9);
  float cosine = dot(P, Q) / denominator;
  return acos(clamp(cosine, -1.0, 1.0));
}

/**
 * Checks the relationship between two sectors and calculates overlap
 * @param a1 First point of sector A
 * @param a2 Second point of sector A
 * @param b1 First point of sector B
 * @param b2 Second point of sector B
 * @param outC1 First intersection point (output)
 * @param outC2 Second intersection point (output)
 * @param outOverlapAngle Overlap angle (output)
 * @return Relationship type (0: disjoint, 1: overlap, 2: contain)
 */
int checkSectorRelation(in vec2 a1, in vec2 a2, in vec2 b1, in vec2 b2,
out vec2 outC1, out vec2 outC2,
out float outOverlapAngle) {
  float a1InB = inMinorArc(b1, b2, a1);
  float a2InB = inMinorArc(b1, b2, a2);
  float b1InA = inMinorArc(a1, a2, b1);
  float b2InA = inMinorArc(a1, a2, b2);

  float aContainedInB = a1InB * a2InB;
  float bContainedInA = b1InA * b2InA;
  float chooseContainA = step(0.5, aContainedInB);
  float chooseContainB = (1.0 - chooseContainA) * step(0.5, bContainedInA);
  float isContain = chooseContainA + chooseContainB;

  float anyAinB = step(0.5, a1InB + a2InB);
  float anyBinA = step(0.5, b1InA + b2InA);
  float overlapCondition = anyAinB * anyBinA;
  float isOverlap = (1.0 - isContain) * overlapCondition;
  float isDisjoint = 1.0 - max(isContain, isOverlap);

  vec2 containC1 = a1 * chooseContainA + b1 * chooseContainB;
  vec2 containC2 = a2 * chooseContainA + b2 * chooseContainB;

  vec2 pickA = mix(a2, a1, a1InB);
  vec2 pickB = mix(b2, b1, b1InA);

  vec2 overlapC1 = pickA * isOverlap;
  vec2 overlapC2 = pickB * isOverlap;

  outC1 = overlapC1 + containC1;
  outC2 = overlapC2 + containC2;

  float length1 = safeLength(outC1);
  float length2 = safeLength(outC2);
  float denominator = max(length1 * length2, 1e-9);
  float cosineValue = dot(outC1, outC2) / denominator;
  float rawAngle = acos(clamp(cosineValue, -1.0, 1.0));
  outOverlapAngle = rawAngle * (1.0 - isDisjoint);

  return int(isOverlap * 1.0 + isContain * 2.0);
}

/**
 * Checks sector relation without calculating loss
 * @param a1 First point of sector A
 * @param a2 Second point of sector A
 * @param b1 First point of sector B
 * @param b2 Second point of sector B
 * @param outC1 First intersection point (output)
 * @param outC2 Second intersection point (output)
 * @return Relationship type (0: disjoint, 1: overlap, 2: contain)
 */
int checkSectorRelationNoLoss(in vec2 a1, in vec2 a2, in vec2 b1, in vec2 b2,
out vec2 outC1, out vec2 outC2) {
  float a1InB = inMinorArc(b1, b2, a1);
  float a2InB = inMinorArc(b1, b2, a2);
  float b1InA = inMinorArc(a1, a2, b1);
  float b2InA = inMinorArc(a1, a2, b2);

  float aContainedInB = a1InB * a2InB;
  float bContainedInA = b1InA * b2InA;
  float chooseContainA = step(0.5, aContainedInB);
  float chooseContainB = (1.0 - chooseContainA) * step(0.5, bContainedInA);
  float isContain = chooseContainA + chooseContainB;

  float anyAinB = step(0.5, a1InB + a2InB);
  float anyBinA = step(0.5, b1InA + b2InA);
  float overlapCondition = anyAinB * anyBinA;
  float isOverlap = (1.0 - isContain) * overlapCondition;
  float isDisjoint = 1.0 - max(isContain, isOverlap);

  vec2 containC1 = a1 * chooseContainA + b1 * chooseContainB;
  vec2 containC2 = a2 * chooseContainA + b2 * chooseContainB;

  vec2 pickA = mix(a2, a1, a1InB);
  vec2 pickB = mix(b2, b1, b1InA);

  vec2 overlapC1 = pickA * isOverlap;
  vec2 overlapC2 = pickB * isOverlap;

  outC1 = overlapC1 + containC1;
  outC2 = overlapC2 + containC2;

  return int(isOverlap * 1.0 + isContain * 2.0);
}

/**
 * Checks if point P is on the segment between A and B
 * @param A First segment endpoint
 * @param B Second segment endpoint
 * @param P Point to check
 * @return True if point is on segment
 */
bool onSegment(vec2 A, vec2 B, vec2 P) {
  vec2 minPoint = min(A, B) - EPSILON;
  vec2 maxPoint = max(A, B) + EPSILON;
  return all(greaterThanEqual(P, minPoint)) && all(lessThanEqual(P, maxPoint));
}

/**
 * Checks if segment from A1 to A2 intersects with segment from origin to B
 * @param A1 First point of first segment
 * @param A2 Second point of first segment
 * @param B Endpoint of second segment
 * @return True if segments intersect
 */
bool segmentsIntersect(vec2 A1, vec2 A2, vec2 B) {
  vec2 origin = vec2(0.0);

  float orientation1 = cross2D(A2 - A1, origin - A1);
  float orientation2 = cross2D(A2 - A1, B - A1);
  float orientation3 = cross2D(B - origin, A1 - origin);
  float orientation4 = cross2D(B - origin, A2 - origin);

  bool generalCase = bool((orientation1 * orientation2 < 0.0f) &&
  (orientation3 * orientation4 < 0.0f));

  bool specialCase =
  bool((abs(orientation1) <= EPSILON && onSegment(A1, A2, origin)) ||
  (abs(orientation2) <= EPSILON && onSegment(A1, A2, B)) ||
  (abs(orientation3) <= EPSILON && onSegment(origin, B, A1)) ||
  (abs(orientation4) <= EPSILON && onSegment(origin, B, A2)));

  return generalCase || specialCase;
}

//////////////////////////////////////////////////////////////
//     Optimized SDF functions
//

/**
 * Signed Distance Function for a box
 * @param point Point to evaluate
 * @param box Box data
 * @return Signed distance to box
 */
float sdfBox(in vec2 point, in BoxData box) {
  vec2 center = (box.rightBottom + box.leftTop) * 0.5;
  vec2 relatedPoint = point - center;
  vec2 halfSize = (box.rightBottom - box.leftTop) * 0.5;

  vec2 d = abs(relatedPoint) - halfSize;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

/**
 * Signed Distance Function for a triangle
 * @param point Point to evaluate
 * @param triangle Triangle data
 * @return Signed distance to triangle
 */
float sdfTriangle(in vec2 point, in TriangleData triangle) {
  vec2 p0 = triangle.point1;
  vec2 p1 = triangle.point2;
  vec2 p2 = triangle.point3;

  vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
  vec2 v0 = point - p0, v1 = point - p1, v2 = point - p2;
  vec2 pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  vec2 pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  vec2 pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
  float s = sign(e0.x * e2.y - e0.y * e2.x);
  vec2 d = min(min(vec2(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
  vec2(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
  vec2(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}

/**
 * Signed Distance Function for a circle
 * @param point Point to evaluate
 * @param circle Circle data
 * @return Signed distance to circle
 */
float sdfCircle(in vec2 point, in CircleData circle) {
  return length(point - circle.center) - circle.radius.x;
}

//////////////////////////////////////////////////////////////
//     Optimized framing points functions
//

/**
 * Gets framing points for a rectangle
 * @param point Evaluation point
 * @param box Box data
 * @return Framing points as vec4 (xy: first point, zw: second point)
 */
vec4 getFramingPointsRectangle(vec2 point, BoxData box) {
  vec2 A = box.leftTop;
  vec2 B = vec2(box.rightBottom.x, box.leftTop.y);
  vec2 C = vec2(box.leftTop.x, box.rightBottom.y);
  vec2 D = box.rightBottom;

  vec2 vecPA = A - point;
  vec2 vecPB = B - point;
  vec2 vecPC = C - point;
  vec2 vecPD = D - point;

  // Precompute angles between vectors
  float angleAB = angleBetween(vecPA, vecPB);
  float angleAC = angleBetween(vecPA, vecPC);
  float angleAD = angleBetween(vecPA, vecPD);
  float angleBC = angleBetween(vecPB, vecPC);
  float angleBD = angleBetween(vecPB, vecPD);
  float angleCD = angleBetween(vecPC, vecPD);

  // Find maximum angle
  float maxAngle = angleAB;
  vec2 result1 = A, result2 = B;

  if (angleAC > maxAngle) {
    maxAngle = angleAC;
    result1 = A;
    result2 = C;
  }
  if (angleAD > maxAngle) {
    maxAngle = angleAD;
    result1 = A;
    result2 = D;
  }
  if (angleBC > maxAngle) {
    maxAngle = angleBC;
    result1 = B;
    result2 = C;
  }
  if (angleBD > maxAngle) {
    maxAngle = angleBD;
    result1 = B;
    result2 = D;
  }
  if (angleCD > maxAngle) {
    maxAngle = angleCD;
    result1 = C;
    result2 = D;
  }

  vec2 vector1 = (result1 - point);
  vec2 vector2 = (result2 - point);
  vector1.y = -vector1.y;
  vector2.y = -vector2.y;

  return vec4(vector1, vector2);
}

/**
 * Gets framing points for a triangle
 * @param point Evaluation point
 * @param triangle Triangle data
 * @return Framing points as vec4 (xy: first point, zw: second point)
 */
vec4 getFramingPointsTriangle(vec2 point, TriangleData triangle) {
  vec2 A = triangle.point1;
  vec2 B = triangle.point2;
  vec2 C = triangle.point3;

  vec2 vecPA = A - point;
  vec2 vecPB = B - point;
  vec2 vecPC = C - point;

  float angleAB = angleBetween(vecPA, vecPB);
  float angleAC = angleBetween(vecPA, vecPC);
  float angleBC = angleBetween(vecPB, vecPC);

  float maxAngle = angleAB;
  vec2 result1 = A, result2 = B;

  if (angleAC > maxAngle) {
    maxAngle = angleAC;
    result1 = A;
    result2 = C;
  }
  if (angleBC > maxAngle) {
    maxAngle = angleBC;
    result1 = B;
    result2 = C;
  }

  return vec4((result1 - point), (result2 - point));
}

/**
 * Gets framing points for a circle
 * @param point Evaluation point
 * @param circle Circle data
 * @return Framing points as vec4 (xy: first point, zw: second point)
 */
vec4 getFramingPointsCircle(vec2 point, CircleData circle) {
  vec2 originToPoint = point - circle.center;
  float distanceSquared = dot(originToPoint, originToPoint);
  float radiusSquared = circle.radius.x * circle.radius.x;

  // Early return if point is inside circle
  if (distanceSquared <= radiusSquared) {
    return vec4(0.0);
  }

  float sqrtDiscriminant = sqrt(distanceSquared - radiusSquared);
  float invDistanceSquared = 1.0 / distanceSquared;

  float dx = originToPoint.x;
  float dy = originToPoint.y;

  // Calculate tangent points
  float tangent1X =
  circle.center.x +
  (radiusSquared * dx - circle.radius.x * dy * sqrtDiscriminant) *
  invDistanceSquared;
  float tangent1Y =
  circle.center.y +
  (radiusSquared * dy + circle.radius.x * dx * sqrtDiscriminant) *
  invDistanceSquared;

  float tangent2X =
  circle.center.x +
  (radiusSquared * dx + circle.radius.x * dy * sqrtDiscriminant) *
  invDistanceSquared;
  float tangent2Y =
  circle.center.y +
  (radiusSquared * dy - circle.radius.x * dx * sqrtDiscriminant) *
  invDistanceSquared;

  vec2 vector1 = vec2(tangent1X, tangent1Y) - point;
  vec2 vector2 = vec2(tangent2X, tangent2Y) - point;
  vector1.y = -vector1.y;
  vector2.y = -vector2.y;

  return vec4(vector1, vector2);
}

void addObj(inout float dist, inout vec3 color, float d, vec3 c) {
    if (dist > d) {
        dist = d;
        color = c;
    }
}

void scene(in vec2 uv, out vec3 color, out float dist) {
    dist=1e9;color=vec3(0,0,0);

    for(int i=0;i<boxCount;++i){
        vec4 col = boxes[i].isLight ? vec4(boxes[i].color*2.,1.):vec4(0.);
        addObj(dist, color, sdfBox(uv,boxes[i]), col.xyz);
    }

    // Process triangles
    for(int i=0;i<triangleCount;++i){
        vec4 col = triangles[i].isLight ? vec4(triangles[i].color*2.,1.) :vec4(0.);
        addObj(dist, color, sdfTriangle(uv,triangles[i]), col.xyz);
    }

    // Process circles
    for(int i=0;i<circleCount;++i){
        vec4 col = circles[i].isLight ? vec4(circles[i].color*2.,1.) : vec4(0.);
        addObj(dist, color, sdfCircle(uv,circles[i]), col.xyz);
    }
}

void trace(vec2 p, vec2 dir, out vec3 c) {
    for (int j = 0; j < 200; j++) {
        float d;
        scene(p, c, d);
        if (d < 1e-3) {
            //c = vec3(0,.1,0);
            return;
        }
        if (d > 1e1) break;
        p -= dir * d;
    }
    c = vec3(0,0,0);
}

float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,
        vec2(12.9898,78.233)))*
        43758.5453123);
}


#define SAMPLES 200

//////////////////////////////////////////////////////////////
//     Main program with optimized performance
//
void main() {
  vec2 uv = fragUV;
  uv.y = 1.0 - uv.y;
  uv.x *= 1920.f / 1080.f;
  uv.x -= 0.4;

  Factor factors[MAX_OBJECTS];
  uint factorCount = 0;

  // Normalized pixel coordinates (from 0 to 1)
    //vec2 uv = fragCoord/iResolution.xy;
    
    vec3 col = vec3(0,0,0);
    for (int i = 0; i < SAMPLES; i++) {
        float t = (float(i) + random(uv+float(i))) / float(SAMPLES) * 2. * 3.1415;
        vec3 c;
        trace(uv, vec2(cos(t), sin(t)), c);
        col += c;
    }
    col /= float(SAMPLES);

    // Output to screen
    fragColor = vec4(col,3.0);

  return;
}