# tree_manager.py
import os
import sys
import sqlite3

TREES_DIR = "trees"
DB_PATH = os.path.join(TREES_DIR, "lattice.db")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def get_domains():
    """Returns a list of all distinct domains in the SQLite database."""
    if not os.path.exists(DB_PATH):
        return []
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT domain_name FROM nodes")
        domains = [row[0] for row in cursor.fetchall()]
    return sorted(domains)


def list_trees():
    clear_screen()
    print("=" * 60)
    print(" AVAILABLE DOMAIN TREES (SQLite DB)")
    print("=" * 60)
    domains = get_domains()
    if not domains:
        print(" No domains found in lattice.db.")
    else:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            for i, domain in enumerate(domains):
                cursor.execute("SELECT COUNT(*) FROM nodes WHERE domain_name = ?", (domain,))
                count = cursor.fetchone()[0]
                print(f" [{i + 1}] Domain: {domain} | Total Nodes: {count}\n")

    input("\nPress Enter to return to menu...")


def inspect_tree():
    clear_screen()
    domains = get_domains()
    if not domains:
        print("No domains available to inspect.")
        input("Press Enter to return...")
        return

    print("Select a domain to inspect:\n")
    for i, d in enumerate(domains):
        print(f" [{i + 1}] {d}")

    try:
        choice = int(input("\nEnter domain number (or 0 to cancel): "))
        if choice == 0:
            return
        selected_domain = domains[choice - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        input("Press Enter to return...")
        return

    clear_screen()
    print(f"--- CELL IDs IN DOMAIN: {selected_domain} ---\n")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cell_id, node_type FROM nodes WHERE domain_name = ?", (selected_domain,))
            cells = cursor.fetchall()
            for i, (cell_id, node_type) in enumerate(cells):
                print(f" {i + 1}. {cell_id} [{str(node_type).upper()}]")
    except Exception as e:
        print(f"Error querying database: {e}")

    input("\nPress Enter to return to menu...")


def main_menu():
    while True:
        clear_screen()
        print("==========================================================")
        print("  STRATA SECURE TREE MANAGER (SQLite Ontology Editor)")
        print("==========================================================")
        print(" [1] List all available domain trees (Overview)")
        print(" [2] Inspect cell IDs of a specific domain")
        print(" [3] Exit")
        print("==========================================================")

        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            list_trees()
        elif choice == "2":
            inspect_tree()
        elif choice == "3":
            clear_screen()
            print("Exiting Tree Manager. Safe travels in the Lattice!")
            sys.exit(0)
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main_menu()
