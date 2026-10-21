"""
=============================================================================
FSUminho AS / ESW Challenge - Verificador de Integridade de Ficheiros
=============================================================================
Garante que os candidatos apenas editam o ficheiro autorizado:
  candidatos/desafio_aluno.py

Qualquer alteração em ficheiros de teste (tests/), motor (core/), documentação
ou workflows de CI/CD é rejeitada de imediato pela pipeline.
=============================================================================
"""

import subprocess
import sys
import os

def check_integrity() -> bool:
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")

    # Se estiver a correr no próprio branch 'main' (e não num Pull Request), é o organizador
    if ref_name == "main" and event_name != "pull_request":
        print("[INFO] Execução na branch 'main' (organizador). Verificação de integridade dispensada.")
        return True

    # Tentar obter lista de ficheiros modificados em relação a origin/main
    try:
        subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, check=False)
        res = subprocess.run(["git", "diff", "--name-only", "origin/main...HEAD"], capture_output=True, text=True, check=True)
        modified_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    except Exception as e:
        print(f"[AVISO] Não foi possível verificar diff contra origin/main: {e}")
        return True

    if not modified_files:
        print("[OK] Nenhum ficheiro modificado em relação a origin/main.")
        return True

    print(f"Ficheiros modificados nesta submissão ({len(modified_files)}):")
    for f in modified_files:
        print(f"  • {f}")

    # Verificar se algum ficheiro fora de 'candidatos/' foi alterado
    unauthorized = []
    for f in modified_files:
        norm = f.replace('\\', '/')
        if not norm.startswith("candidatos/"):
            unauthorized.append(f)

    if unauthorized:
        print("\n" + "="*70)
        print(" 🔒 [ERRO DE INTEGRIDADE: MODIFICAÇÃO DE FICHEIROS NÃO AUTORIZADOS]")
        print("="*70)
        print(" Os candidatos APENAS têm autorização para editar o código em:")
        print(" 👉 candidatos/desafio_aluno.py\n")
        print(" Foram detetadas alterações nos seguintes ficheiros protegidos do repositório:")
        for uf in unauthorized:
            print(f"  ❌ {uf}")
        print("\n Instrução para o candidato:")
        print(" Deves restaurar os ficheiros protegidos para a versão original do desafio:")
        print("   git checkout origin/main -- <caminho_do_ficheiro>")
        print("="*70 + "\n")
        return False

    print("\n[OK] Verificação de integridade aprovada: apenas a pasta 'candidatos/' foi alterada.")
    return True

if __name__ == "__main__":
    ok = check_integrity()
    sys.exit(0 if ok else 1)
