# Desafio Técnico (Sênior) — Classificação de Legibilidade de Documentos (v3)

## Contexto

Em fluxos que dependem do envio de documentos pelo usuário (documentos de identificação, contratos, procurações, comprovantes, etc.), imagens de baixa qualidade — desfocadas, mal enquadradas, escuras, com reflexo ou cortadas — geram retrabalho, atrito na jornada do usuário e risco de erro nas etapas seguintes do pipeline (OCR, extração de dados, validação).

Você deve projetar e construir um componente de controle de qualidade que atue na entrada desse pipeline: um modelo que recebe a imagem de um documento textual e decide se ela está em condições de leitura (**legível**) ou não (**ilegível**).

## Objetivo

Conduzir o ciclo completo de desenvolvimento de um modelo de classificação binária de legibilidade de documentos, da definição do problema à apresentação de resultados, como se este componente fosse entrar em produção em um sistema real.

### Definição operacional de legibilidade

Para que todos os candidatos trabalhem com o mesmo critério de saída, adotamos a seguinte definição:

> Uma imagem de documento é considerada **legível** quando é possível recuperar pelo menos **80% das informações textuais impressas** presentes no documento. Informações manuscritas não entram nesse cálculo — apenas o conteúdo impresso deve ser considerado na avaliação da legibilidade.

Essa definição deve orientar suas decisões ao longo de todo o projeto, especialmente na construção/rotulagem do dataset e na escolha da estratégia de avaliação.

## Escopo e Restrições

- Linguagem: Python. Frameworks e ferramentas são de livre escolha — justifique as decisões relevantes.
- Uso de modelos pré-treinados é permitido.
- Não é necessário treinar em GPU própria.
- Entrega em repositório Git (ou notebook + scripts), código executável e reprodutível.

## O que se espera

Não há um dataset pronto e rotulado para este problema específico. Parte do desafio é sua capacidade de investigar, tomar decisões de engenharia de dados e justificá-las tecnicamente — isso é parte do que está sendo avaliado, não um obstáculo a ser contornado com uma receita.

Da mesma forma, não há uma arquitetura, métrica de corte ou estratégia de avaliação "correta" pré-definida. Espera-se que você tome essas decisões de forma independente e as sustente com argumentos técnicos, considerando o contexto de negócio descrito acima.

O projeto deve cobrir, com a profundidade que você julgar apropriada para demonstrar domínio sênior:

1. **Dados** — obtenção/construção de uma base adequada ao problema, com documentação de fontes, licenças e decisões tomadas.
2. **Análise exploratória** — entendimento do dado, dos rótulos e de eventuais vieses e limitações.
3. **Modelagem** — ao menos uma baseline e um modelo principal, com justificativa de escolhas.
4. **Avaliação** — métricas apropriadas ao problema, considerando as implicações operacionais de diferentes tipos de erro.
5. **Apresentação** — um relatório claro, que comunique metodologia, resultados, limitações e próximos passos para diferentes públicos (técnico e não técnico).

## Entregáveis

1. Código-fonte organizado e comentado.
2. Relatório de resultados (Markdown, PDF ou notebook renderizado).
3. Instruções de reprodução do experimento (ambiente, dependências, comandos).
4. Opcional: uma forma simples de demonstrar o modelo em uso (CLI, API mínima, interface simples).

## O que NÃO é o foco do desafio

Não é esperado um modelo state-of-the-art. O processo, as decisões técnicas e a capacidade de argumentar sobre trade-offs importam mais do que a métrica final isolada.
