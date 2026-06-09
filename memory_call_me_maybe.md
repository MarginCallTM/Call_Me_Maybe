# memory_call_me_maybe.md — Contexte de passation du projet « Call Me Maybe »

> **À quoi sert ce fichier.** C'est une passation de contexte entre deux postes
> de travail (Lenovo de l'École 42 ↔ MacBook M2 perso). Quand je reprends le
> projet depuis chez moi, je demande à Claude de **lire ce fichier en entier**
> puis d'en **intégrer le contenu utile à sa propre mémoire** (`MEMORY.md` /
> fichiers de mémoire). Il complète — il ne remplace pas — `CLAUDE.md`,
> `TODO.txt` et `en.subject.pdf`, qui restent les sources de référence.
>
> Dernière mise à jour : **2026-06-09**.

---

## 0. Posture et workflow pédagogique (À RESPECTER ABSOLUMENT)

- Je suis un **étudiant de l'École 42**, solide en C, débutant en Python avancé
  et **parti de zéro sur le fonctionnement interne des LLM** (tokenisation,
  logits, décodage). Ce projet est autant un exercice d'**apprentissage** que de
  technique.
- **Claude guide, ne fait pas le travail à ma place.** Le workflow est :
  1. Je pose une question / demande une implémentation.
  2. Claude **génère le code dans le CLI** (dans sa réponse), bien commenté et
     expliqué.
  3. **Je retape le code à la main** dans VSCode (volontaire, pédagogique).
  4. Claude **n'écrit JAMAIS directement dans les fichiers**, SAUF si je le
     demande explicitement (« édite le fichier X », « corrige directement »…).
- **Langue** : explications **en français**, code (commentaires, docstrings,
  noms) **en anglais**.
- Format de réponse attendu : résumé en 2-3 phrases → explication du concept
  (en démystifiant si c'est une notion LLM) → code annoté à retaper → récap des
  points clés + pièges flake8/mypy/sujet.

---

## 1. But du projet

Transformer un **prompt en langage naturel** en un **appel de fonction
structuré et 100 % valide**, via **constrained decoding fait main**.

Exemple :
- Entrée : `"What is the sum of 2 and 3?"`
- Sortie : `{ "prompt": "...", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0} }`

**Principe central** : on ne fait PAS confiance au modèle pour produire du bon
JSON. On génère le JSON **token par token** et, à chaque étape, on met à
**`-inf`** les logits des tokens interdits. Le JSON est donc valide **par
construction**, pas par chance.

**Sortie finale** : un seul fichier `data/output/function_calling_results.json`,
tableau JSON, un objet par prompt, clés **exactement** `prompt`, `name`,
`parameters`. `data/output/` ne doit PAS être commité.

---

## 2. Concepts LLM (rappels — je pars de zéro, répéter aide)

- **Token** : un morceau de texte (souvent un bout de mot). Analogie : des
  briques Lego réutilisables.
- **Tokenisation** : texte → liste de tokens. Le tokenizer ajoute un symbole
  d'espace `Ġ` (U+0120) devant un mot précédé d'une espace.
- **Input IDs** : chaque token a un entier unique. Le modèle ne voit QUE des
  nombres.
- **Logits** : pour CHAQUE token du vocabulaire, le modèle donne un score
  (avant normalisation). Plus haut = plus probable comme prochain token.
- **Décodage greedy** : choisir le token au plus haut score (`argmax`).
- **Constrained decoding** (cœur du projet) : avant de choisir, on met les
  logits des tokens interdits à `-inf`, donc le modèle NE PEUT PAS produire
  d'invalide. On lui retire physiquement les mauvaises options de la main.
- **Deux décisions distinctes à chaque étape** :
  - **Quels tokens sont autorisés** (le masque) = décidé par NOUS, selon la
    POSITION dans la grammaire JSON + le schéma de la fonction. **Indépendant du
    prompt.**
  - **Lequel des autorisés on prend** (l'argmax) = décidé par le MODÈLE, via ses
    logits influencés par le prompt.
  - Slogan : **« le masquage garantit la STRUCTURE, le modèle apporte le
    CONTENU. »**
- **Pourquoi partir de TOUT à `-inf` puis rouvrir les autorisés** (liste
  blanche, « deny by default ») plutôt que blacklister les interdits :
  1. défaut sûr (un oubli = token indisponible, pas un JSON cassé) ;
  2. on sait calculer l'ensemble des AUTORISÉS via `vocab.ids_matching` ;
  3. les tokens spéciaux/padding (sans texte) sont à `-inf` d'office.
- **Pourquoi `-inf` et pas un petit nombre** : `-inf` est par définition < à
  tout réel, donc un token interdit perd TOUJOURS l'argmax. Aucune faille.

---

## 3. Contraintes techniques OBLIGATOIRES (sujet)

- **Python ≥ 3.10**, gestionnaire **`uv`** (le correcteur lance `uv sync`).
- Dépendances autorisées : **`numpy`** et **`pydantic`** uniquement.
- Code dans `src/`, lancé via `uv run python -m src`. `llm_sdk/` copié à la
  racine, au même niveau que `src/`.
- **pydantic** pour TOUTE validation (`BaseModel`, champs typés, validation
  auto).
- **flake8** strict (lignes ≤ 79, pas d'import inutile, espacement PEP8).
- **mypy** : type hints partout, docstrings (Google/NumPy) sur classes/méthodes.
- Gestion **gracieuse** des erreurs : jamais de crash, message clair,
  try/except + context managers.

### INTERDICTIONS (très important)
- ❌ `dspy`, `pytorch`/`torch`, `huggingface`/`transformers`, `outlines`, et
  tout package haut niveau similaire **dans mon code `src/`**. Le constrained
  decoding est fait **à la main**.
  - ⚠️ Le SDK fourni (`llm_sdk`) importe torch/transformers en interne : ça,
    c'est **toléré** (package fourni). C'est MON code qui ne doit pas les
    importer.
  - ⚠️ `huggingface_hub.hf_hub_download` est utilisé seulement dans des
    **scripts de test jetables** (non rendus) pour récupérer le chemin de
    `vocab.json` sans charger le modèle. À ne PAS laisser dans `src/`.
- ❌ Méthodes/attributs **privés** de `llm_sdk` (préfixe `_` : `_model`,
  `_tokenizer`, `_device`…).
- ❌ Choisir la fonction par **heuristique** (regex, mots-clés, if/else sur le
  texte). Le choix DOIT venir des logits du modèle.
- ❌ Compter sur le **prompt** pour produire du JSON valide → la fiabilité vient
  du **constrained decoding**.
- ❌ **Hardcoder** les exemples (fonctions/prompts) : les fichiers d'entrée
  CHANGENT à la correction. Recalculer les IDs au runtime depuis `vocab.json`.

### Modèle
- Modèle par défaut OBLIGATOIRE : **Qwen/Qwen3-0.6B**.

---

## 4. API RÉELLE de `llm_sdk` (≠ doc du sujet / CLAUDE.md — coder contre la VRAIE)

La doc (PDF, CLAUDE.md, TODO.txt) ne correspond PAS à la vraie API de
`Small_LLM_Model` (dans `llm_sdk/llm_sdk/__init__.py`). Utiliser les VRAIES
signatures :

- `get_logits_from_input_ids(input_ids: list[int]) -> list[float]`
  → prend une **liste**, renvoie une **liste de floats** (PAS un Tensor). Donc
  le masquage se fait sur une liste Python ; numpy est un choix (retenu pour la
  vitesse), torch est inutile.
- `get_path_to_vocab_file() -> str` (PAS `get_path_to_vocabulary_json()`).
- `encode(text: str) -> torch.Tensor` (2-D ; `.tolist()[0]` pour avoir
  `list[int]`).
- `decode(token_ids) -> str` (conforme à la doc).
- Bonus dispo : `get_path_to_merges_file()`, `get_path_to_tokenizer_file()`.

---

## 5. Faits tokenizer Qwen3-0.6B (établis en Phase 1)

- **Taille des logits = 151936**, mais `vocab.json` n'a que **151643** entrées.
  → les IDs en trop sont spéciaux/padding, SANS texte, à masquer à `-inf`.
  Utiliser `id_to_token.get(i)` (renvoie `None` si absent), JAMAIS `[i]`.
- Chiffres `0`–`9` = IDs **15..24** (bloc contigu). `'0'`=15 … `'9'`=24.
- Caractères JSON 1-char : `{`=90  `}`=92  `[`=58  `]`=60  `"`=1  `:`=25
  `,`=11  `.`=13  `-`=12.
- Booléens : `true`=1866, `false`=3849 (chacun **un seul token**).
- Espace : n'existe que **collé** via `Ġ` (U+0120) = 220 ; pas de token
  espace-nu.
- **Tokens combinés multi-caractères** : `{"`=4913  `":`=788  `":"`=3252
  `",`=497  `},`=2137  `"}`=9207. → « 1 char = 1 token » est **FAUX** : un token
  peut faire avancer le JSON de plusieurs caractères.
- **Décisions de design** : générer du JSON **compact** (sans espaces) pour
  éviter `Ġ` ; **ne jamais hardcoder** les IDs, les recalculer au runtime.
- Les tokens byte-level ne se concatènent PAS comme du texte (espaces = `Ġ`,
  octets remappés) → pour retrouver le vrai texte d'une string, utiliser
  `decode()` du SDK.

---

## 6. Environnement & setup

- `uv 0.11.7`, Python 3.13.1 en local (le projet exige `>=3.10`).
- `numpy>=2.0.0` (2.4.6 résolu) + `pydantic` (2.13.4) dans `pyproject.toml`.
- `llm_sdk` câblé comme **membre de workspace uv** (import OK).
- ⚠️ `llm_sdk` tire **torch + CUDA** → `.venv` ≈ **4.7 G**. Instancier
  `Small_LLM_Model()` charge le modèle 0.6B **entier en RAM** (lourd, a déjà
  fait planter le PC une fois). Cache HF ≈ 1.5 G au premier chargement.
- `.gitignore` ignore `__pycache__`, `.mypy_cache`, `.venv`, `data/output/`.

### Travail multi-machines (42 Lenovo x86 Linux ↔ MacBook M2 ARM)
- Le **code source** (`src/`) est **100 % portable** : logits, numpy, machines à
  états, pydantic, JSON ne dépendent ni de la marque, ni de l'OS, ni du CPU.
  Il n'y a **AUCUNE** « optimisation pour Lenovo » à faire (c'est un
  faux problème).
- La **seule** vraie différence Mac/Linux = l'**environnement**, surtout
  **`torch`** (binaires ARM ≠ x86). `uv` gère ça.
- Règles :
  - Synchroniser par **git** (le **code seulement**). NE JAMAIS transporter le
    `.venv` (il est gitignoré). Faire `uv sync` **sur chaque machine**.
  - Le cache modèle (~1.5 G) se retéléchargera une fois sur le Mac.
  - Vérifier que `uv sync` passe sur le Mac (torch parfois capricieux sur
    macOS-ARM). En cas d'échec, refaire `uv lock` pour réintégrer la plateforme
    Mac.
  - **RÈGLE D'OR** : lancer la passe complète (`uv sync` + `make run` +
    `make lint`) **sur un Lenovo de 42 au moins une fois avant le rendu**. Les
    logits peuvent varier infinitésimalement entre builds torch ARM/x86
    (flottants/BLAS) → en théorie un choix de token pourrait rarement basculer.
    Valider accuracy/temps sur la plateforme cible. (Le M2 est probablement
    plus rapide → objectif « < 5 min » confortable.)

---

## 7. AVANCEMENT (phases) — au 2026-06-09

- ✅ **Phase 0** — Environnement (uv, deps, `llm_sdk`, arborescence,
  `.gitignore`, `uv sync` OK, fichiers d'entrée en place).
- ✅ **Phase 1** — Compréhension du terrain (API réelle du SDK, structure
  `vocab.json`, IDs clés, pièges `Ġ` et tokens multi-char).
- ✅ **Phase 2** — `src/models.py` : 4 modèles pydantic.
- ✅ **Phase 3** — `src/io_handler.py` : I/O JSON + erreurs.
- ✅ **Phase 4** — `src/tokenizer_utils.py` : classe `Vocabulary` (sauf bonus
  4.5 encode/decode à la main).
- 🔶 **Phase 5** — Constrained decoding (EN COURS), découpée en 4 couches :
  - ✅ **Couche 0** : masquage + argmax (`mask_logits`, `select_next_token`).
  - ✅ **Couche 1** : générer UNE valeur typée — **number, string, boolean**
    (les 3 testées, OK le 2026-06-09).
  - ⬜ **Couche 2** : machine à états de l'**objet complet**
    `{ "clé": val, ... }` qui ORCHESTRE les 3 générateurs. ← **REPRENDRE ICI**
  - ⬜ **Couche 3** : garde-fou (max tokens) + condition d'arrêt globale.
- ⬜ **Phase 6** — Choix de la fonction **par le LLM** (pas heuristique) :
  contraindre le champ `name` aux seuls noms existants.
- ⬜ **Phase 7** — `src/pipeline.py` : orchestration prompt → FunctionCall.
- ⬜ **Phase 8** — `src/cli.py` + `src/__main__.py` (args
  `--functions_definition`, `--input`, `--output`).
- ⬜ **Phase 9** — Qualité (flake8, mypy avec les flags du sujet).
- ⬜ **Phase 10** — Makefile (install/run/debug/clean/lint/lint-strict).
- ⬜ **Phase 11** — Tests. **Phase 12** — README. **Phase 13** — Finalisation.

---

## 8. Code déjà écrit dans `src/` (retapé à la main, lints OK)

- `src/__init__.py` — vide (pour `python -m src`).
- `src/models.py` — 4 `BaseModel`, tous `ConfigDict(extra="forbid")` :
  - `ParameterSpec(type: Literal["number","string","boolean"])`
  - `FunctionDefinition(name, description, parameters: dict[str, ParameterSpec],
    returns: ParameterSpec)`
  - `Prompt(prompt: str)`
  - `FunctionCall(prompt, name, parameters: dict[str, float|str|bool])`
  - NB : `parameters` = descripteurs de TYPE en entrée, VALEURS en sortie.
    Piège connu : `bool` est sous-classe de `int`.
- `src/io_handler.py` — exception maison `DataError` (approche A : tout
  emballer, un seul `except` au point d'entrée). `_read_json` (with open +
  json.load, catch OSError + JSONDecodeError), `load_functions` /
  `load_prompts` (TypeAdapter(list[...]) + ValidationError), `write_results`
  (Path.mkdir(parents, exist_ok) + json.dump indent=2 ensure_ascii=False).
  `raise ... from exc` partout.
- `src/tokenizer_utils.py` — `SPACE_MARKER="Ġ"` + classe `Vocabulary` (SIMPLE,
  PAS pydantic, par décision délibérée ; prend un chemin str, découplée du
  modèle lourd). Tables `token_to_id` + `id_to_token` (inverse). Méthodes :
  `_load` (réutilise DataError), `__len__`, `token_for_id(id) -> str|None`,
  `id_of(tok) -> int|None`, `ids_matching(predicate) -> set[int]` (LE pont vers
  le constrained decoding : on passe une règle, on reçoit les IDs autorisés),
  `is_space_prefixed(tok)`.
- `src/constrained_decoder.py` — voir détail §9.

### Scripts de test JETABLES (racine, NON rendus, NON dans src/)
- `explore.py`, `test.py` (démo with-open), `test_vocab.py` (récupère le chemin
  de vocab.json via `hf_hub_download` SANS charger le modèle), `test_mask.py`
  (faux logits : un token interdit à gros logit n'est jamais choisi),
  `test_number.py`, `test_string.py`, `test_boolean.py`.
- ⚠️ Ces tests importent `huggingface_hub` → à NE PAS confondre avec `src/`.
  Penser à les retirer/ignorer avant le rendu final.

---

## 9. `src/constrained_decoder.py` — détail (Phase 5, Couches 0 + 1)

**Design commun** : `get_logits` est **injecté** (un `Callable`, pas le modèle)
→ découplage, testable sans charger le modèle lourd, `src/` reste sans torch.
Les tests jetables passent un faux `get_logits` scénarisé (logit 100 au token
voulu), donc AUCUN chargement de modèle.

- `class DecodeError(Exception)` — état impossible du décodeur (miroir de
  `DataError`).
- `mask_logits(logits, allowed_ids) -> np.ndarray` — part d'un tableau tout à
  `-inf`, rouvre seulement les IDs autorisés (indexation vectorisée).
- `select_next_token(logits, allowed_ids) -> int` — argmax sur le masque =
  meilleur token PARMI les autorisés ; lève `DecodeError` si l'ensemble est
  vide.
- `generate_number(get_logits, prefix_ids, vocab, stop_ids, max_tokens=32)
  -> str` + helpers `_number_specials`, `_allowed_for_number`,
  `_next_number_state` + constante `_DIGITS`.
  - Machine à 5 états : `start` → (`int_lead`) → `int` → (`frac_lead`) →
    `frac`.
  - `stop_ids` = tokens autorisés à SUIVRE le nombre (ex. `}`, `,`), offerts
    SEULEMENT dans les états complets (`int`, `frac`) → le modèle décide la
    longueur mais avec des fins LÉGALES. Doit être **non vide**.
  - `for/else` = garde-fou anti-boucle (max_tokens). Renvoie une `str`.
  - Prédicat chiffres réutilisé de Phase 1 : `t != "" and all(c in _DIGITS
    for c in t)` (accepte les multi-chiffres « 12 »).
- `generate_string(get_logits, decode, prefix_ids, vocab, max_tokens=64)
  -> str` + prédicat `_is_string_content`.
  - **Auto-délimitée** : le `"` fermant EST le stop (PAS de `stop_ids`).
  - `_is_string_content` interdit tout token contenant `"` ou `\` → JSON valide
    garanti. **Limite** : pas d'échappements pour l'instant (à revoir pour la
    fonction regex).
  - `decode` est **injecté** (`model.decode`) car les tokens byte-level (`Ġ`,
    octets remappés) ne se concatènent pas.
  - Suit DEUX listes : `generated` (envoyée au modèle, AVEC les 2 guillemets) vs
    `content` (décodée → la valeur, SANS les guillemets). Renvoie la valeur sans
    guillemets.
- `generate_boolean(get_logits, prefix_ids, vocab) -> bool`.
  - UN seul token parmi `{true_id, false_id}` (recalculés via `id_of`, PAS
    hardcodés). Ni état, ni boucle. Renvoie un `bool` (`token_id == true_id`).
  - Le choix vient du MODÈLE, pas d'une heuristique (règle du sujet).

---

## 10. LEÇONS & PIÈGES rencontrés (à ne pas réapprendre)

- **La grammaire (masque) gagne TOUJOURS sur le favori du modèle.** Un favori
  interdit → `-inf` → ignoré. Si les autorisés restants sont à égalité (faux
  logits tous à 0.0), `np.argmax` prend le **plus petit ID** (ex. chiffre `'0'`
  = id 15 ; entre `true`/`false`, `true`=1866 < `false`=3849 → True).
- **Un 2e `.` dans un nombre est impossible PAR CONSTRUCTION** (état `frac`
  n'autorise pas le point). Idem une lettre dans un nombre. C'est le constrained
  decoding qui marche.
- **Piège des tests** : `stop_ids = {script[-1]}` → modifier la liste du script
  change le token d'arrêt sans s'en rendre compte.
- **Piège Python** : deux littéraux string accolés `"o" "b"` sont concaténés par
  Python en `"ob"`.
- **Piège PEP 263** : un commentaire dans les 2 PREMIÈRES lignes d'un `.py`
  contenant `coding:` ou `coding=` est pris pour une déclaration d'encodage →
  `SyntaxError "unknown encoding"`. Ne jamais écrire `coding:` en tête de
  fichier.
- **flake8 fréquents** : 2 lignes vides entre fonctions top-level (E302),
  espaces traînants (W291), ligne vide entre groupes d'imports
  (stdlib / third-party / local).

---

## 11. Données d'entrée actuelles (CHANGERONT à la correction — ne pas hardcoder)

5 fonctions dans `data/input/functions_definition.json` :
- `fn_add_numbers` [a, b : number]
- `fn_greet` [name : string]
- `fn_reverse_string` [s : string]
- `fn_get_square_root` [a : number]
- `fn_substitute_string_with_regex` [source_string, regex, replacement :
  string] ← **le cas le plus dur** (3 strings dont une regex, guillemets
  imbriqués, échappements) → c'est ce qui motivera l'évolution de
  `generate_string`.

+ 11 prompts de test dans `data/input/function_calling_tests.json`.

---

## 12. PROCHAINE ÉTAPE — Couche 2

Construire la **machine à états de l'objet complet** dans
`src/constrained_decoder.py` :
`START → "{" → clé (string, parmi les params attendus du schéma) → ":" →
valeur (selon le type du param : appelle generate_number / generate_string /
generate_boolean) → "," (param suivant) ou "}" (fin) → END`.

Elle **orchestre** les 3 générateurs de la Couche 1. Les `stop_ids` d'un nombre
deviennent alors naturellement `{ "," , "}" }`. Contraintes de type tirées de
`functions_definition.json`. Puis Couche 3 (garde-fou global + arrêt), puis
Phase 6 (choix de la fonction par le LLM).
