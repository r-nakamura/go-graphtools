package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	. "github.com/r-nakamura/go-graphtools"
)

var (
	directed   = flag.Bool("d", true, "directed graph (default)")
	format     = flag.String("i", "dot", "input graph format")
	stats      = flag.String("t", "", "specify statistic to display (nodes/edges/degree/dstddev/dmin/dmax/direct/connect/clcoeff/pathlen/maxcomp)")
	undirected = flag.Bool("u", false, "undirected graph")
	verbose    = flag.Bool("v", false, "verbose mode")
)

var (
	statsMap = map[string]interface{} {
		"nodes":   func(g *Graph) float64 { return float64(len(g.Vertices())) },
		"edges":   func(g *Graph) float64 { return float64(len(g.Edges())) },
		"degree":  func(g *Graph) float64 { return float64(g.AverageDegree()) },
		"dstddev": func(g *Graph) float64 {
			d := make([]float64, len(g.Vertices()))
			for i, val := range g.GraphAttribute("degree").([]int) {
				d[i] = float64(val)
			}
			return math.Sqrt(variance(d))
		},
		"dmin":    func(g *Graph) float64 {
			d := g.GraphAttribute("degree").([]int)
			sort.Ints(d)
			return float64(d[0])
		},
		"dmax":    func(g *Graph) float64 {
			d := g.GraphAttribute("degree").([]int)
			sort.Ints(d)
			return float64(d[len(d)-1])
		},
		"direct":  func(g *Graph) float64 {
			if g.Directed() {
				return 1
			} else {
				return 0
			}
		},
		"connect": func(g *Graph) float64 {
			if g.IsConnected() {
				return 1
			} else {
				return 0
			}
		},
		"clcoeff": func(g *Graph) float64 { return clusteringCoefficient(g) },
		"pathlen": func(g *Graph) float64 {
			g.FloydWarshall()
			var total, count float64
			for _, u := range g.Vertices() {
				for _, v := range g.Vertices() {
					if u == v {
						continue
					}
					if _, ok := g.T[u][v]; !ok {
						continue
					}
					total += g.T[u][v]
					count++
				}
			}
			return total / count
		},
		"maxcomp": func(g *Graph) float64 { return float64(len(g.MaximalComponent())) },
	}
)

func mean(alist []float64) float64 {
	var sum float64
	for _, val := range alist {
		sum += val
	}
	return sum / float64(len(alist))
}

func variance(alist []float64) float64 {
	m := mean(alist)
	for i, val := range alist {
		alist[i] = math.Pow((val - m), 2)
	}
	return mean(alist)
}

func clusteringCoefficient(g *Graph) float64 {
	var total, count float64
	for _, v := range g.Vertices() {
		degree := float64(g.Degree(v))
		if degree < 2 {
			continue
		}

		var m float64
		for _, u := range g.Neighbors(v) {
			for _, w := range g.Neighbors(v) {
				if u < w {
					continue
				}
				if g.HasEdge(u, w) || g.HasEdge(w, u) {
					m++
				}
			}
		}
		total += m / ((degree * (degree - 1)) / 2)
		count++
	}
	return total / count
}

func formatNumber(val float64) string {
	if val < math.Pow(10, 10) {
		return fmt.Sprintf("%14.4f", val)
	} else {
		return fmt.Sprintf("%14.4g", val)
	}
}

func readInput(file string) ([]string, error) {
	var f *os.File
	var err error
	if file == "" {
		f = os.Stdin
	} else {
		f, err = os.Open(file)
		if err != nil {
			return nil, err
		}
		defer f.Close()
	}

	scanner := bufio.NewScanner(f)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	return lines, err
}

func main() {
	flag.Parse()

	var file string
	if args := flag.Args(); len(args) > 0 {
		file = args[0]
	} else {
		file = ""
	}

	var g Graph
	g.New()
	if *directed {
		g.SetDirected()
	}
	if *undirected {
		g.SetUndirected()
	}
	lines, err := readInput(file)
	if err != nil {
		os.Exit(1)
	}
	g.ImportGraph(*format, lines)

	degree := make([]int, len(g.Vertices()))
	for i, v := range g.Vertices() {
		degree[i] = g.Degree(v)
	}
	g.SetGraphAttribute("degree", degree)

	var statsType []string
	if *stats != "" {
		statsType = strings.Split(*stats, ",")
	} else {
		statsType = []string{
			"nodes", "edges", "degree", "dstddev", "dmin", "dmax",
			"direct", "connect", "clcoeff",
		}
	}
	for _, atype := range statsType {
		if _, ok := statsMap[atype]; ok {
			val := statsMap[atype].(func(*Graph) float64)(&g)
			if *verbose {
				fmt.Printf("%s\t", atype)
			}
			fmt.Println(formatNumber(val))
		} else {
			fmt.Printf("no support for '%s'\n", atype)
			os.Exit(1)
		}
	}
}
