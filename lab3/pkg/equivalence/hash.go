package equivalence

import (
	"bst-equivalence/pkg/bst"
)

func ComputeHash(bst2compute *bst.BinarySearchTree) int {
	if bst2compute.InOrderTraversal == nil {
		// Populate the in-order traversal if not already done
		bst2compute.InOrderTraversal = []int{}
		bst.InOrderTraversalBST(bst2compute, &bst2compute.InOrderTraversal)
	}

	hash := 1
	for _, value := range bst2compute.InOrderTraversal {
		newValue := value + 2
		hash = (hash*newValue + newValue) % 1000
	}
	return hash
}
