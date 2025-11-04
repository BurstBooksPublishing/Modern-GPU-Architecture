#include <vector>
// patternMask: bitmask for pattern cells; tileW,tileH tile dimensions.
std::vector<std::pair<int,int>> expandPattern(uint32_t patternMask,int tileW,int tileH){
    std::vector<std::pair<int,int>> offsets;
    for(int y=0;y<tileH;y++){
        for(int x=0;x<tileW;x++){
            if(patternMask & (1u<<(y*tileW+x))){
                offsets.emplace_back(x,y);
            }
        }
    }
    return offsets;
}