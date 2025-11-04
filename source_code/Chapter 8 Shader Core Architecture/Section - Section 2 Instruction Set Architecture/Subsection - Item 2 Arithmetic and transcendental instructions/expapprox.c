#include 
// Approximate exp(x) with range reduction and 5-term minimax poly.
// Uses fma for single rounding per operation, mapping to hardware FMA.
float exp_approx(float x) {
    const float ln2 = 0.69314718056f;
    const float inv_ln2 = 1.44269504089f;
    // Range reduction: k = round(x/ln2)
    int k = (int)floorf(x * inv_ln2 + 0.5f);
    float r = x - k * ln2;
    // Polynomial coefficients (example minimax coefficients)
    const float c4 = 0.001328f, c3 = 0.008333f;
    const float c2 = 0.041667f, c1 = 0.166667f, c0 = 1.0f;
    // Horner evaluation with fma to map to single FMA instructions
    float y = fmaf(c4, r, c3);
    y = fmaf(y, r, c2);
    y = fmaf(y, r, c1);
    y = fmaf(y, r, c0);
    // Reconstruct: result = ldexpf(y, k) i.e., y * 2^k
    return ldexpf(y, k); // maps to fast shift/multiply
}