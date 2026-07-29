# data_augmentation

Essa pasta reúne os scripts que a gente usa pra gerar variações "sujas" das imagens originais: tipo simular fotos tiradas com câmera ruim, mal iluminadas, borradas, amassadas, com manchas etc. A ideia é aumentar a base de dados pra treinar modelos mais robustos a esse tipo de ruído do mundo real.

Tem três formas de gerar essas variações, cada uma com um script separado, porque cada lib ataca o problema de um jeito diferente.

## Antes de rodar qualquer coisa

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Copie `.env.example` para `.env` e preencha com suas próprias chaves:

   ```bash
   cp .env.example .env
   ```

   ```
   OPENAI_API_KEY=sua-chave-aqui
   HUGGINGFACE_TOKEN=seu-token-aqui
   ```

   O `.env` já está no `.gitignore`, então não tem risco de subir chave pro repositório sem querer. **Nunca** cole a chave direto no código ou em arquivo `.txt` solto na pasta — já rolou isso aqui antes e não é legal.

3. Coloque as imagens originais em `../database/input` (ou seja, uma pasta `database/input` no nível acima de `data_augmentation/`). Só `.jpg`, `.jpeg` e `.png` são reconhecidos.

## Os scripts

### `generate_database_albumentations.py`

Usa a lib [Albumentations](https://albumentations.ai/) pra aplicar ~30 transformações clássicas de visão computacional: blur, ruído, mudanças de brilho/contraste, distorção de perspectiva, sombra, chuva, neve, compressão JPEG, etc. É rápido, roda tudo localmente, sem custo de API.

```bash
python generate_database_albumentations.py
```

Saída vai pra `../database/albumentations/`, com um arquivo por combinação imagem × transformação (`nome-da-imagem_NomeDaTransformacao.jpg`).

### `generate_database_augraphy.py`

Usa a lib [Augraphy](https://augraphy.readthedocs.io/), que é focada em simular efeitos de **documentos escaneados/impressos**: fotocópia ruim, manchas de tinta, dobras, furos de encadernação, moiré, watermark, etc. Também roda 100% local.

```bash
python generate_database_augraphy.py
```

Saída em `../database/augraphy/`.

### `generate_database_openai.py`

Esse aqui é diferente: usa o modelo `gpt-image-2` da OpenAI pra editar as imagens com prompts em linguagem natural (escurecer, clarear, simular câmera ruim, glare, crop). Como envolve chamadas de API **pagas**, o script:

- avisa antes de começar quantas chamadas vai fazer (arquivos × 5 configurações);
- pede confirmação no terminal antes de disparar tudo;
- tenta de novo (até 3x, com backoff) se alguma chamada falhar, em vez de simplesmente quebrar no meio do lote.

```bash
python generate_database_openai.py
```

Dá uma olhada no [preço atual da API](https://openai.com/api/pricing) antes de rodar em uma base grande — é fácil perder a noção do custo total quando o script multiplica arquivos por configurações. Saída em `../database/openai/`.

## Estrutura gerada

Cada script escreve na sua própria subpasta dentro de `../database/`, pra não misturar tudo:

```
database/
├── input/              <- suas imagens originais (você que cria)
├── albumentations/      <- saída do script 1
├── augraphy/             <- saída do script 2
└── openai/               <- saída do script 3
```

Cada pasta de saída também ganha um arquivo `.log` (ex: `albumentations.log`) com o histórico completo da execução — útil pra saber depois quais imagens falharam e por quê, sem precisar rolar o terminal pra cima.

## `common.py`

Não é pra rodar direto — é um módulo com as funções que os três scripts compartilham (achar as imagens de entrada, criar pastas de saída, configurar o log, carregar o `.env`). Se for mexer em algo estrutural (tipo mudar onde as pastas ficam), é aqui.

## Coisas pra ficar de olho

- Os scripts pulam imagem que falhar (erro vai pro log) e continuam o processamento, é preciso conferir o  `.log` no final pra ver se nada ficou faltando.
- O script da OpenAI é o único que custa dinheiro de verdade. Os outros dois são de graça, só usam CPU/GPU local.
