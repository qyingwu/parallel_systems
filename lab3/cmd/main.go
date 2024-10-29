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
		hash2bstId = semaphoreImpl(bstList, hashWorkers, dataWorkers)
	} else {
		log.Fatalf("Invalid combination of hash-workers and data-workers")
	}

	hashTime := time.Since(startHash).Seconds()
	fmt.Printf("hashTime: %.8f\n", hashTime)

	// Print hashes with associated BST IDs
	startGroup := time.Now()
	for hash, ids := range hash2bstId {
		if len(ids) > 1 {
			fmt.Printf("%d: %s\n", hash, joinIntSlice(ids, " "))
		}
	}
	hashGroupTime := time.Since(startGroup).Seconds()
	fmt.Printf("hashGroupTime: %.8f\n", hashGroupTime)

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
		// parallel comparison with buffering and chunking
		var comparisons [][2]int
		totalTrees := len(bstList)

		// Pre-allocate the comparisons slice with calculated capacity
		maxComparisons := 0
		for _, ids := range hash2bstId {
			n := len(ids)
			if n > 1 {
				maxComparisons += (n * (n - 1)) / 2
			}
		}
		comparisons = make([][2]int, 0, maxComparisons)

		// Collect all comparisons needed
		for _, ids := range hash2bstId {
			if len(ids) > 1 {
				for i := 0; i < len(ids); i++ {
					for j := i + 1; j < len(ids); j++ {
						comparisons = append(comparisons, [2]int{ids[i], ids[j]})
					}
				}
			}
		}

		// Dynamically calculate optimal chunk size based on number of workers and comparisons
		chunkSize := max(100, len(comparisons)/(compWorkers*4))
		numChunks := (len(comparisons) + chunkSize - 1) / chunkSize
		chunks := make(chan [][2]int, numChunks)

		// Split comparisons into chunks
		for i := 0; i < len(comparisons); i += chunkSize {
			end := min(i+chunkSize, len(comparisons))
			chunks <- comparisons[i:end]
		}
		close(chunks)

		// Use a sharded UnionFind to reduce lock contention
		shardedUF := make([]*ufs.UnionFind, compWorkers)
		for i := range shardedUF {
			shardedUF[i] = ufs.NewUnionFind(totalTrees)
		}

		// Process chunks in parallel with local UnionFind instances
		var wgComp sync.WaitGroup
		for i := 0; i < compWorkers; i++ {
			wgComp.Add(1)
			workerID := i
			go func() {
				defer wgComp.Done()
				localUF := shardedUF[workerID]
				for chunk := range chunks {
					for _, pair := range chunk {
						id1, id2 := pair[0], pair[1]
						if equivalence.CompareBST(bstList[id1], bstList[id2]) {
							localUF.Union(id1, id2)
						}
					}
				}
			}()
		}
		wgComp.Wait()

		// Merge results from all shards
		for i := 1; i < len(shardedUF); i++ {
			for j := 0; j < totalTrees; j++ {
				if shardedUF[i].Find(j) != j {
					uf.Union(j, shardedUF[i].Find(j))
				}
			}
		}
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
	taskCh := make(chan int, len(bstList))
	resultCh := make(chan [2]int, hashWorkers)

	// Start hash workers
	var hashWg sync.WaitGroup
	for i := 0; i < hashWorkers; i++ {
		hashWg.Add(1)
		go func() {
			defer hashWg.Done()
			for idx := range taskCh {
				tree := bstList[idx]
				tree.Hash = equivalence.ComputeHash(tree)
				resultCh <- [2]int{idx, tree.Hash}
			}
		}()
	}

	// Start data workers
	var mapWg sync.WaitGroup
	for i := 0; i < dataWorkers; i++ {
		mapWg.Add(1)
		go func() {
			defer mapWg.Done()
			for result := range resultCh {
				idx, hash := result[0], result[1]
				hash2bstId[hash] = append(hash2bstId[hash], idx)
			}
		}()
	}

	// Send tasks to hash workers
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

	for i := 0; i < len(bstList); i++ {
		tasks <- i
	}
	close(tasks)
	wg.Wait()

	return hash2bstId
}

// Sharded map structure with fine-grained locking on each shard
type ShardedMap struct {
	shards    []map[int][]int
	locks     []sync.Mutex
	numShards int
}

func NewShardedMap(numShards int) *ShardedMap {
	shards := make([]map[int][]int, numShards)
	locks := make([]sync.Mutex, numShards)
	for i := range shards {
		shards[i] = make(map[int][]int)
	}
	return &ShardedMap{
		shards:    shards,
		locks:     locks,
		numShards: numShards,
	}
}

// getShard returns the shard index based on the hash
func (sm *ShardedMap) getShard(hash int) int {
	return hash % sm.numShards
}

func (sm *ShardedMap) Append(hash, value int) {
	shardIndex := sm.getShard(hash)
	sm.locks[shardIndex].Lock() // Lock only the specific shard
	defer sm.locks[shardIndex].Unlock()

	sm.shards[shardIndex][hash] = append(sm.shards[shardIndex][hash], value)
}

// Merge combines all shards into a single map for the final result
func (sm *ShardedMap) Merge() map[int][]int {
	result := make(map[int][]int)
	for _, shard := range sm.shards {
		for hash, values := range shard {
			result[hash] = append(result[hash], values...)
		}
	}
	return result
}

func semaphoreImpl(bstList []*bst.BinarySearchTree, hashWorkers, dataWorkers int) map[int][]int {
	shardedMap := NewShardedMap(dataWorkers)
	// Allows up to `dataWorkers` goroutines
	semaphore := make(chan struct{}, dataWorkers)

	var wg sync.WaitGroup
	tasks := make(chan int, len(bstList))

	// Start hash workers
	for i := 0; i < hashWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range tasks {
				tree := bstList[idx]
				tree.Hash = equivalence.ComputeHash(tree)

				semaphore <- struct{}{}
				shardedMap.Append(tree.Hash, idx)
				<-semaphore
			}
		}()
	}

	for i := 0; i < len(bstList); i++ {
		tasks <- i
	}
	close(tasks)
	wg.Wait()

	return shardedMap.Merge()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
