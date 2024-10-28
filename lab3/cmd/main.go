package main

import (
	"bst-equivalence/pkg/bst"
	"bst-equivalence/pkg/equivalence"
	"bst-equivalence/pkg/ufs"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Command-line flags
var (
	hashWorkers int
	dataWorkers int
	compWorkers int
	inputFile   string
)

func init() {
	flag.IntVar(&hashWorkers, "hash-workers", 1, "Number of goroutines for hashing BSTs")
	flag.IntVar(&dataWorkers, "data-workers", 1, "Number of goroutines for updating the map")
	flag.IntVar(&compWorkers, "comp-workers", 1, "Number of goroutines for tree comparisons")
	flag.StringVar(&inputFile, "input", "simple.txt", "Input file path")
}

func main() {
	flag.Parse()
	data, err := os.ReadFile(inputFile)
	if err != nil {
		log.Fatalf("Failed to read input file: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")

	// Step 1: Construct BSTs
	bstList := make([]*bst.BinarySearchTree, len(lines))
	for i, line := range lines {
		values := convertToIntSlice(strings.Fields(line))
		bstList[i] = &bst.BinarySearchTree{Id: i}
		bst.BuildBST(values, bstList[i])
	}

	// Step 2: Group BSTs by hash using selected implementation
	var hash2bstId map[int][]int
	startHash := time.Now()

	if hashWorkers == 1 && dataWorkers == 1 {
		// Case 1: Fully sequential implementation
		hash2bstId = sequentialImpl(bstList)
	} else if dataWorkers == 1 && hashWorkers > 1 {
		// Case 2: Parallel hashing with central manager using a channel
		hash2bstId = channelImpl(bstList, hashWorkers, dataWorkers)
	} else if hashWorkers == dataWorkers && hashWorkers > 1 {
		// Case 3: Each hashing goroutine updates the map individually with a single mutex
		hash2bstId = mutexImpl(bstList, hashWorkers, dataWorkers)
	} else if hashWorkers > dataWorkers && dataWorkers > 1 {
		// Case 4 (Optional): i hash workers, j data workers with fine-grained control
		hash2bstId = shardedImpl(bstList, hashWorkers, dataWorkers)
	} else {
		log.Fatalf("Invalid combination of hash-workers and data-workers")
	}

	hashTime := time.Since(startHash).Seconds()
	fmt.Printf("hashTime: %.8f\n", hashTime)

	// Print hashes with associated BST IDs (excluding single-tree groups)
	for hash, ids := range hash2bstId {
		if len(ids) > 1 {
			fmt.Printf("%d: %s\n", hash, joinIntSlice(ids, " "))
		}
	}

	// Step 3: Parallelize tree comparisons
	startCompare := time.Now()
	uf := ufs.NewUnionFind(len(bstList))

	if compWorkers == 1 {
		// Sequential comparison
		for _, ids := range hash2bstId {
			if len(ids) > 1 {
				for i := 0; i < len(ids); i++ {
					for j := i + 1; j < len(ids); j++ {
						isEqual := equivalence.CompareBST(bstList[ids[i]], bstList[ids[j]])
						if isEqual {
							uf.Union(ids[i], ids[j])
						}
					}
				}
			}
		}
	} else {
		// Parallel comparison with multiple workers
		compCh := make(chan [2]int, len(hash2bstId)*len(bstList))
		var wgComp sync.WaitGroup
		for i := 0; i < compWorkers; i++ {
			wgComp.Add(1)
			go func() {
				defer wgComp.Done()
				for pair := range compCh {
					id1, id2 := pair[0], pair[1]
					isEqual := equivalence.CompareBST(bstList[id1], bstList[id2])
					if isEqual {
						uf.Union(id1, id2)
					}
				}
			}()
		}

		for _, ids := range hash2bstId {
			if len(ids) > 1 {
				for i := 0; i < len(ids); i++ {
					for j := i + 1; j < len(ids); j++ {
						compCh <- [2]int{ids[i], ids[j]}
					}
				}
			}
		}

		close(compCh)
		wgComp.Wait()
	}

	compareTreeTime := time.Since(startCompare).Seconds()
	fmt.Printf("compareTreeTime: %.8f\n", compareTreeTime)

	// Group equivalent trees and print
	groups := make(map[int][]int)
	for i := 0; i < len(bstList); i++ {
		root := uf.Find(i)
		groups[root] = append(groups[root], i)
	}

	groupNum := 0
	for _, group := range groups {
		if len(group) > 1 {
			fmt.Printf("group %d: %s\n", groupNum, joinIntSlice(group, " "))
			groupNum++
		}
	}
}

func joinIntSlice(ints []int, sep string) string {
	strs := make([]string, len(ints))
	for i, v := range ints {
		strs[i] = strconv.Itoa(v)
	}
	return strings.Join(strs, sep)
}

func convertToIntSlice(values []string) []int {
	intSlice := make([]int, len(values))
	for i, v := range values {
		num, err := strconv.Atoi(v)
		if err != nil {
			log.Fatalf("Error converting string to int: %v", err)
		}
		intSlice[i] = num
	}
	return intSlice
}

func sequentialImpl(bstList []*bst.BinarySearchTree) map[int][]int {
	hash2bstId := make(map[int][]int)
	for _, tree := range bstList {
		tree.Hash = equivalence.ComputeHash(tree)
		hash2bstId[tree.Hash] = append(hash2bstId[tree.Hash], tree.Id)
	}
	return hash2bstId
}

// channel-based implementation
func channelImpl(bstList []*bst.BinarySearchTree, hashWorkers, dataWorkers int) map[int][]int {
	hash2bstId := make(map[int][]int)
	hashLocks := make(map[int]*sync.Mutex)
	var hashLocksMu sync.Mutex

	taskCh := make(chan int, len(bstList))
	resultCh := make(chan [2]int, len(bstList))

	var hashWg sync.WaitGroup
	for i := 0; i < hashWorkers; i++ {
		hashWg.Add(1)
		go func() {
			defer hashWg.Done()
			for idx := range taskCh {
				tree := bstList[idx]
				tree.Hash = equivalence.ComputeHash(tree)

				hashLocksMu.Lock()
				if _, exists := hashLocks[tree.Hash]; !exists {
					hashLocks[tree.Hash] = &sync.Mutex{}
				}
				hashLocksMu.Unlock()

				resultCh <- [2]int{idx, tree.Hash}
			}
		}()
	}

	var mapWg sync.WaitGroup
	for i := 0; i < dataWorkers; i++ {
		mapWg.Add(1)
		go func() {
			defer mapWg.Done()
			for result := range resultCh {
				idx, hash := result[0], result[1]
				hashLocksMu.Lock()
				var hashLock = hashLocks[hash]
				hashLocksMu.Unlock()

				hashLock.Lock()
				hash2bstId[hash] = append(hash2bstId[hash], idx)
				hashLock.Unlock()
			}
		}()
	}

	for i := 0; i < len(bstList); i++ {
		taskCh <- i
	}
	close(taskCh)
	hashWg.Wait()
	close(resultCh)
	mapWg.Wait()

	return hash2bstId
}

// Single mutex implementation
func mutexImpl(bstList []*bst.BinarySearchTree, hashWorkers, dataWorkers int) map[int][]int {
	hash2bstId := make(map[int][]int)
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Create worker pool
	tasks := make(chan int, len(bstList))
	for i := 0; i < hashWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range tasks {
				tree := bstList[idx]
				tree.Hash = equivalence.ComputeHash(tree)
				mu.Lock()
				hash2bstId[tree.Hash] = append(hash2bstId[tree.Hash], idx)
				mu.Unlock()
			}
		}()
	}

	// Send tasks
	for i := 0; i < len(bstList); i++ {
		tasks <- i
	}
	close(tasks)
	wg.Wait()

	return hash2bstId
}

// Sharded mutex implementation
type ShardedMap struct {
	shards    []map[int][]int
	locks     []sync.Mutex
	numShards int
}

func NewShardedMap(numShards int) *ShardedMap {
	sm := &ShardedMap{
		shards:    make([]map[int][]int, numShards),
		locks:     make([]sync.Mutex, numShards),
		numShards: numShards,
	}
	for i := range sm.shards {
		sm.shards[i] = make(map[int][]int)
	}
	return sm
}

func (sm *ShardedMap) getShard(hash int) int {
	return hash % sm.numShards
}

func (sm *ShardedMap) Append(hash, value int) {
	shard := sm.getShard(hash)
	sm.locks[shard].Lock()
	sm.shards[shard][hash] = append(sm.shards[shard][hash], value)
	sm.locks[shard].Unlock()
}

func (sm *ShardedMap) Merge() map[int][]int {
	result := make(map[int][]int)
	for _, shard := range sm.shards {
		for hash, values := range shard {
			result[hash] = append(result[hash], values...)
		}
	}
	return result
}

func shardedImpl(bstList []*bst.BinarySearchTree, hashWorkers, dataWorkers int) map[int][]int {
	shardedMap := NewShardedMap(dataWorkers) // Use dataWorkers as number of shards
	var wg sync.WaitGroup

	// Create worker pool
	tasks := make(chan int, len(bstList))
	for i := 0; i < hashWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range tasks {
				tree := bstList[idx]
				tree.Hash = equivalence.ComputeHash(tree)
				shardedMap.Append(tree.Hash, idx)
			}
		}()
	}

	// Send tasks
	for i := 0; i < len(bstList); i++ {
		tasks <- i
	}
	close(tasks)
	wg.Wait()

	return shardedMap.Merge()
}
