#include <stdio.h>

double pcie_bw_gbps(int lanes, double gt_s, double encoding_eff) {
    // gt_s: GT/s per lane, encoding_eff e.g. 0.8 or 128.0/130.0
    double bits_per_sec = lanes * gt_s * 1.0 * encoding_eff; // bits/s
    double bytes_per_sec = bits_per_sec / 8.0;
    return bytes_per_sec / 1e9; // GB/s
}

int main(void) {
    printf("PCIe3 x16 ~ %.3f GB/s\n", pcie_bw_gbps(16, 8.0e9, 128.0/130.0)); // example
    printf("PCIe4 x16 ~ %.3f GB/s\n", pcie_bw_gbps(16, 16.0e9, 128.0/130.0));
    printf("PCIe5 x16 ~ %.3f GB/s\n", pcie_bw_gbps(16, 32.0e9, 128.0/130.0));
    return 0;
}