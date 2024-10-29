package bst

// TreeNode represents a node in a binary tree
type TreeNode struct {
	Val    int
	Left   *TreeNode
	Right  *TreeNode
	Parent *TreeNode
}

type BinarySearchTree struct {
	Id               int
	Root             *TreeNode
	Hash             int
	InOrderTraversal []int
}

func insertIntoBST(bst *BinarySearchTree, val int) *TreeNode {
	if bst.Root == nil {
		bst.Root = &TreeNode{Val: val}
		return bst.Root
	}
	return insert(bst.Root, val)
}

func insert(node *TreeNode, val int) *TreeNode {
	if node == nil {
		return &TreeNode{Val: val}
	}
	if val < node.Val {
		node.Left = insert(node.Left, val)
	} else {
		node.Right = insert(node.Right, val)
	}
	return node
}

func BuildBST(values []int, bst *BinarySearchTree) {
	for _, val := range values {
		insertIntoBST(bst, val)
	}
}

func InOrderTraversalBST(bst *BinarySearchTree, result *[]int) {
	inOrderTraversal(bst.Root, result)
}

// In-order traversal helper function
func inOrderTraversal(node *TreeNode, result *[]int) {
	if node == nil {
		return
	}
	inOrderTraversal(node.Left, result)
	*result = append(*result, node.Val)
	inOrderTraversal(node.Right, result)
}
