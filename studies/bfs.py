graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E', 'F'],
  'C' : ['G'],
  'D' : [],
  'E' : [],
  'F' : ['H'],
  'G' : ['I'],
  'H' : [],
  'I' : []
}

def bfs(graph, node):
    visited = [] # estrutura para salvar quem ja visitei, evitando ciclos
    fifo = [] # fila (queue)

    visited.append(node)
    fifo.append(node)

    while fifo:
        s = fifo.pop(0) # pega o primeiro
        print(s, end = ' ')

        for n in graph[s]: # para todos os filhos desse no atual
            if n not in visited: # se eu nao visitei ainda
                visited.append(n)
                fifo.append(n)

def main():
    bfs(graph, 'A')

main()