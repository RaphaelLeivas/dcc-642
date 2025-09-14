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

def dfs(graph, node):
    visited = [] # estrutura para salvar quem ja visitei, evitando ciclos
    lifo = [] # pilha (stack)

    visited.append(node)
    lifo.append(node)

    while lifo:
        s = lifo.pop() # pega o topo da stack
        print(s, end = ' ')

        for n in graph[s]: # para todos os filhos desse no atual
            if n not in visited: # se eu nao visitei ainda
                visited.append(n)
                lifo.append(n)

def main():
    dfs(graph, 'A')

main()