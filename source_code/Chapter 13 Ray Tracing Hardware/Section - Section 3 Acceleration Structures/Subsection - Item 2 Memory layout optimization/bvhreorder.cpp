#include 
struct Node { float bmin[3], bmax[3]; int left, right; int firstPrim, primCount; };
void flattenDFS(const Node* root, std::vector& flat) {
  flat.reserve(1024); // pre-reserve to reduce reallocations
  std::function dfs = [&](const Node* n)->int{
    int idx = flat.size();
    flat.push_back(*n); // placeholder copy; indices fixed after recursion
    if (n->left >= 0) { // internal node
      int leftIdx = dfs(&root[n->left]); // flatten left subtree
      int rightIdx = dfs(&root[n->right]); // flatten right subtree
      flat[idx].left = leftIdx; flat[idx].right = rightIdx; // update children
    }
    return idx;
  };
  dfs(root);
}