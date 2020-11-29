# go-graphtools

go-graphtools - tools for graph theory and network science with many generation models

# DESCRIPTION

This module is a golang implementation of graph-tools written in
Python language.

# SYNOPSIS

```go
package main
import (
	"fmt"

    "github.com/r-nakamura/go-graphtools"
)

func main() {
	var g DirectedGraph
	g.New()
	g.CreateRandomGraph(10, 20)
	fmt.Println(g.ExportDot())
}
```

# INSTALLATION

```sh
go get github.com/r-nakamura/go-graphtools
```

# SEE ALSO

graph-tools - [https://github.com/h-ohsaki/graph-tools](https://github.com/h-ohsaki/graph-tools)

# AUTHOR

Ryo Nakamura <nakamura[atmark]zebulun.net>
