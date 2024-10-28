package equivalence

import (
	"bst-equivalence/pkg/bst"
)

func CompareBST(bst1 *bst.BinarySearchTree, bst2 *bst.BinarySearchTree) bool {
	// Check if hashes differ, indicating trees are not equivalent
	if bst1.Hash != bst2.Hash {
		return false
	}

	return compareBST(bst1, bst2)
}

func compareBST(bst1 *bst.BinarySearchTree, bst2 *bst.BinarySearchTree) bool {

	if bst1.InOrderTraversal == nil {
		bst1.InOrderTraversal = []int{}
		bst.InOrderTraversalBST(bst1, &bst1.InOrderTraversal)
	}
	if bst2.InOrderTraversal == nil {
		bst2.InOrderTraversal = []int{}
		bst.InOrderTraversalBST(bst2, &bst2.InOrderTraversal)
	}

	if len(bst1.InOrderTraversal) != len(bst2.InOrderTraversal) {
		return false
	}
	for i := 0; i < len(bst1.InOrderTraversal); i++ {
		if bst1.InOrderTraversal[i] != bst2.InOrderTraversal[i] {
			return false
		}
	}
	return true
}
