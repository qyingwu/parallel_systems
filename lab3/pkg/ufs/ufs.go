package ufs

import (
	"sync"
)

// Union-Find data structure
type UnionFind struct {
	parent []int
	rank   []int
	locks  map[int]*sync.Mutex
	mu     sync.Mutex
}

func NewUnionFind(size int) *UnionFind {
	parent := make([]int, size)
	rank := make([]int, size)
	for i := range parent {
		parent[i] = i
	}
	return &UnionFind{
		parent: parent,
		rank:   rank,
		locks:  make(map[int]*sync.Mutex),
	}
}

func (uf *UnionFind) getLock(id int) *sync.Mutex {
	uf.mu.Lock()
	defer uf.mu.Unlock()
	if lock, exists := uf.locks[id]; exists {
		return lock
	}
	lock := &sync.Mutex{}
	uf.locks[id] = lock
	return lock
}

func (uf *UnionFind) Find(x int) int {
	if uf.parent[x] != x {
		uf.parent[x] = uf.Find(uf.parent[x]) // Path compression
	}
	return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) {
	// Acquire locks in a consistent order to prevent deadlocks
	if x > y {
		x, y = y, x
	}
	lockX := uf.getLock(x)
	lockY := uf.getLock(y)

	lockX.Lock()
	lockY.Lock()
	defer lockX.Unlock()
	defer lockY.Unlock()

	rootX := uf.Find(x)
	rootY := uf.Find(y)
	if rootX != rootY {
		if uf.rank[rootX] < uf.rank[rootY] {
			uf.parent[rootX] = rootY
		} else if uf.rank[rootX] > uf.rank[rootY] {
			uf.parent[rootY] = rootX
		} else {
			uf.parent[rootY] = rootX
			uf.rank[rootX]++
		}
	}
}
