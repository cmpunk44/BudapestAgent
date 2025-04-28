# FONTOS: Először állítsuk be a környezeti változókat, bármilyen import előtt
import os
import sys

# API kulcsok beállítása közvetlenül a környezeti változókban
os.environ["OPENAI_API_KEY"] = "xxx"
os.environ["MAPS_API_KEY"] = "xxx"

# Ezután importáljuk a többi modult
import time
import csv
import json
from datetime import datetime

# Most importáljuk a LangChain és a többi modult
try:
    from langchain_openai import ChatOpenAI
    from agent import tools, prompt, Agent
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.errors import GraphRecursionError
except ImportError as e:
    print(f"Hiba a függőségek importálásakor: {e}")
    print("Győződj meg arról, hogy telepítetted a szükséges csomagokat:")
    print("pip install langchain-openai langchain-core langgraph")
    exit(1)

# Tesztfeladatok definiálása
test_cases = [
    # Alapfeladatok
    {
        "id": "basic_route_1",
        "category": "basic",
        "query": "Hogyan juthatok el a Keleti pályaudvarról a Budai Várba?"
    },
    {
        "id": "basic_attraction_1",
        "category": "basic",
        "query": "Mi az a Lánchíd?"
    },
    {
        "id": "basic_restaurant_1",
        "category": "basic",
        "query": "Mutass éttermeket a Váci utca közelében."
    },
    
    # Összetett feladatok
    {
        "id": "complex_route_1",
        "category": "complex",
        "query": "Szeretnék eljutni a Hősök teréről a Parlamenthez, majd onnan a Budai Várhoz. Útközben szeretnék ebédelni valahol."
    },
    {
        "id": "complex_thematic_1",
        "category": "complex",
        "query": "Szeretnék egy történelmi látványosságokat bemutató útitervet a Deák térről indulva, ami 4 órát vesz igénybe."
    },
    {
        "id": "complex_special_1",
        "category": "complex",
        "query": "Mutass egy útitervet, ami kerüli a zsúfolt helyeket, és főként szabadtéri látnivalókat tartalmaz."
    },
    
    # Szélsőséges esetek
    {
        "id": "edge_nonexistent_1",
        "category": "edge",
        "query": "Hogyan juthatok el a Keleti pályaudvarról a Nem Létező Múzeumba?"
    },
    {
        "id": "edge_long_1",
        "category": "edge",
        "query": "Szeretnék egy részletes útitervet, amely Budapesten a következő helyszíneket tartalmazza kronológiai sorrendben: Keleti pályaudvar, Nemzeti Múzeum, Váci utca, Vörösmarty tér, Duna-part, Lánchíd, Budai Vár, Halászbástya, Mátyás-templom, Gellért-hegy, Citadella, Szabadság-szobor, Margit-sziget, Parlamentet, és a végén szeretnék egy jó vacsorázó helyet találni, amely autentikus magyar ételeket kínál. Útközben szeretnék ebédelni valahol a Budai Vár környékén, lehetőleg terasszal rendelkező étteremben. A teljes útiterv 8 órában férjen bele, és részletes leírást szeretnék minden látnivalóról, külön kiemelve azok történelmi jelentőségét."
    },
    {
        "id": "edge_contradiction_1",
        "category": "edge",
        "query": "Szeretnék egy 2 órás útitervet, amely tartalmazza az összes fontos budapesti látnivalót, és részletesen elmagyarázza mindegyiket."
    },
    {
        "id": "edge_gibberish_1",
        "category": "edge",
        "query": "Qwerty xyzabc deák tér múzeum látnivaló?"
    }
]

# Ágens konfigurációk létrehozása
def create_agent_with_model(model_name="gpt-4o-mini", use_tools=True):
    """Adott modellel és eszközkészlettel hoz létre egy ágenst"""
    model = ChatOpenAI(model=model_name, openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.3)
    
    if use_tools:
        return Agent(model, tools, system=prompt)
    else:
        # Módosított prompt az eszközök nélküli használathoz
        no_tools_prompt = prompt.replace("using parse_input_tool", "by thinking about").replace(
            "call directions_tool", "consider").replace("use attractions_tool", "think about places")
        return Agent(model, [], system=no_tools_prompt)

# Eredmények naplózására szolgáló osztály
class TestLogger:
    def __init__(self, filename=None):
        """Inicializálja a naplózót"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.csv"
            self.new_file = True
        else:
            self.new_file = not os.path.exists(filename)
        
        self.filename = filename
        self.results = []
        
        # CSV fejléc létrehozása ha új fájl
        if self.new_file:
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'TestID', 'Category', 'Configuration', 'Query', 
                    'ResponseTime', 'ToolCalls', 'Success', 'ErrorType',
                    'Accuracy', 'Completeness', 'Usability', 
                    'Notes', 'Response'
                ])
    
    def log_result(self, test_id, category, config, query, response_time, 
                  tool_calls=0, success=True, error_type="", 
                  accuracy=None, completeness=None, usability=None, 
                  notes="", response=""):
        """Eredmény hozzáadása a naplóhoz"""
        result = {
            'TestID': test_id,
            'Category': category,
            'Configuration': config,
            'Query': query,
            'ResponseTime': response_time,
            'ToolCalls': tool_calls,
            'Success': success,
            'ErrorType': error_type,
            'Accuracy': accuracy,
            'Completeness': completeness,
            'Usability': usability,
            'Notes': notes,
            'Response': response
        }
        
        self.results.append(result)
        
        # Eredmény mentése CSV-be
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                result['TestID'], result['Category'], result['Configuration'], 
                result['Query'], result['ResponseTime'], result['ToolCalls'],
                result['Success'], result['ErrorType'],
                result['Accuracy'], result['Completeness'], result['Usability'],
                result['Notes'], result['Response'][:2000]  # Csak az első 500 karakter
            ])
        
        return result
    
    def generate_summary(self):
        """Összefoglaló statisztikák generálása"""
        if not self.results:
            return "Nincsenek eredmények a statisztikához."
        
        # Manuális statisztika számítás
        configs = {}
        categories = {}
        total_tests = len(self.results)
        
        for result in self.results:
            config = result['Configuration']
            category = result['Category']
            response_time = result['ResponseTime']
            success = result['Success']
            
            # Konfigurációk szerinti csoportosítás
            if config not in configs:
                configs[config] = {'count': 0, 'total_time': 0, 'success_count': 0, 'tool_calls': 0}
            
            configs[config]['count'] += 1
            configs[config]['total_time'] += response_time
            configs[config]['tool_calls'] += result['ToolCalls']
            if success:
                configs[config]['success_count'] += 1
            
            # Kategóriák szerinti csoportosítás
            if category not in categories:
                categories[category] = {config: {'count': 0, 'total_time': 0, 'success_count': 0} for config in configs.keys()}
            
            if config not in categories[category]:
                categories[category][config] = {'count': 0, 'total_time': 0, 'success_count': 0}
                
            categories[category][config]['count'] += 1
            categories[category][config]['total_time'] += response_time
            if success:
                categories[category][config]['success_count'] += 1
        
        # Statisztikák számítása
        avg_response_times = {}
        success_rates = {}
        avg_tool_calls = {}
        
        for config, data in configs.items():
            if data['count'] > 0:
                avg_response_times[config] = data['total_time'] / data['count']
                success_rates[config] = (data['success_count'] / data['count']) * 100
                avg_tool_calls[config] = data['tool_calls'] / data['count']
            else:
                avg_response_times[config] = 0
                success_rates[config] = 0
                avg_tool_calls[config] = 0
        
        # Kategóriánkénti statisztikák
        category_stats = {}
        for category, config_data in categories.items():
            category_stats[category] = {}
            for config, data in config_data.items():
                if data['count'] > 0:
                    category_stats[category][config] = {
                        'avg_time': data['total_time'] / data['count'],
                        'success_rate': (data['success_count'] / data['count']) * 100
                    }
                else:
                    category_stats[category][config] = {
                        'avg_time': 0,
                        'success_rate': 0
                    }
        
        # Eredmény szótár összeállítása
        summary = {
            'total_tests': total_tests,
            'avg_response_time': avg_response_times,
            'success_rates': success_rates,
            'avg_tool_calls': avg_tool_calls,
            'category_stats': category_stats
        }
        
        return summary

# Eszközhívások számlálása a válaszból
def count_tool_calls(result):
    """Megszámolja az eszközhívások számát a válaszban"""
    tool_calls = 0
    for message in result.get("messages", []):
        if hasattr(message, 'tool_calls'):
            tool_calls += len(message.tool_calls)
    return tool_calls

# Teszt végrehajtó függvény
def run_tests(continue_from=None):
    """Végrehajtja a teszteket és naplózza az eredményeket"""
    # Logger inicializálása
    logger = TestLogger(continue_from)
    
    # Ágens konfigurációk létrehozása
    configs = {
        "GPT-4o with tools": create_agent_with_model("gpt-4o", True),
        "GPT-4o-mini with tools": create_agent_with_model("gpt-4o-mini", True),
        "GPT-4o no tools": create_agent_with_model("gpt-4o", False)
    }
    
    # Tesztek végrehajtása
    for test_case in test_cases:
        print(f"\nFuttatás: {test_case['id']} - {test_case['query'][:50]}...")
        
        for config_name, agent in configs.items():
            print(f"  Konfiguráció: {config_name}")
            
            # Mérjük az időt
            start_time = time.time()
            
            try:
                # Futtatjuk az ágenst
                result = agent.graph.invoke(
                    {"messages": [HumanMessage(content=test_case["query"])]},
                    {"recursion_limit": 15}  # Növelt recursion limit
                )
                
                # Számoljuk az időt
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"    Válaszidő: {elapsed_time:.2f} másodperc")
                
                # Eszközhívások számolása
                tool_calls = count_tool_calls(result)
                print(f"    Eszközhívások: {tool_calls}")
                
                # Válasz kinyerése
                response_content = ""
                if result["messages"] and len(result["messages"]) > 0:
                    response_content = result["messages"][-1].content
                
                # Eredmény naplózása - sikeres
                logger.log_result(
                    test_id=test_case["id"],
                    category=test_case["category"],
                    config=config_name,
                    query=test_case["query"],
                    response_time=elapsed_time,
                    tool_calls=tool_calls,
                    success=True,
                    response=response_content
                )
                
            except GraphRecursionError as e:
                # Recursion limit hiba esetén
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"    HIBA: Recursion limit túllépve: {str(e)}")
                
                # Eredmény naplózása - recursion hiba
                logger.log_result(
                    test_id=test_case["id"],
                    category=test_case["category"],
                    config=config_name,
                    query=test_case["query"],
                    response_time=elapsed_time,
                    tool_calls=0,
                    success=False,
                    error_type="RecursionError",
                    notes=f"Recursion limit hiba: {str(e)}",
                    response="A model túllépte a megengedett eszközhívások számát."
                )
                
            except Exception as e:
                # Egyéb hibák esetén
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"    HIBA: {type(e).__name__}: {str(e)}")
                
                # Eredmény naplózása - egyéb hiba
                logger.log_result(
                    test_id=test_case["id"],
                    category=test_case["category"],
                    config=config_name,
                    query=test_case["query"],
                    response_time=elapsed_time,
                    tool_calls=0,
                    success=False,
                    error_type=type(e).__name__,
                    notes=f"Hiba: {str(e)}",
                    response="Hiba történt a feldolgozás során."
                )
    
    # Statisztika létrehozása és kiírása
    print("\nTesztelés befejezve!")
    summary = logger.generate_summary()
    print("\nÖsszesítő statisztikák:")
    
    print(f"  Összes teszt: {summary['total_tests']}")
    
    print("  Átlagos válaszidő:")
    for config, avg_time in summary['avg_response_time'].items():
        print(f"    {config}: {avg_time:.2f} másodperc")
    
    print("  Sikerességi arány:")
    for config, rate in summary['success_rates'].items():
        print(f"    {config}: {rate:.1f}%")
    
    print("  Átlagos eszközhívások száma:")
    for config, avg_calls in summary['avg_tool_calls'].items():
        print(f"    {config}: {avg_calls:.2f}")
    
    print("  Kategóriánkénti statisztikák:")
    for category, config_data in summary['category_stats'].items():
        print(f"    {category}:")
        for config, stats in config_data.items():
            print(f"      {config}:")
            print(f"        Átlagos válaszidő: {stats['avg_time']:.2f} másodperc")
            print(f"        Sikerességi arány: {stats['success_rate']:.1f}%")
    
    print(f"\nA részletes eredmények itt érhetők el: {logger.filename}")
    
    return logger

# Manual accuracy evaluation helper
def manual_evaluation():
    """Segédfüggvény az eredmények manuális értékeléséhez"""
    # CSV fájl meghatározása
    results_file = input("Add meg a CSV fájl nevét (default: legutolsó eredményfájl): ")
    
    if not results_file:
        # Legutóbbi fájl keresése
        files = [f for f in os.listdir('.') if f.startswith('test_results_') and f.endswith('.csv')]
        if not files:
            print("Nem található eredményfájl!")
            return
        results_file = max(files)  # A legutolsó fájl (időbélyeg alapján)
    
    print(f"Fájl: {results_file}")
    
    # CSV olvasása
    rows = []
    with open(results_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Hiányzó értékelések keresése
    unevaluated = [row for row in rows if not row.get('Accuracy')]
    
    if len(unevaluated) == 0:
        print("Minden teszteset értékelve van!")
        return
    
    print(f"{len(unevaluated)} értékeletlen teszteset van.")
    
    # Tesztesetek értékelése
    updated_rows = []
    for row in rows:
        if row.get('Accuracy'):  # Ha már értékelve van, nem módosítjuk
            updated_rows.append(row)
            continue
        
        # Csak sikeres teszteket értékeljünk
        if row.get('Success') == 'False':
            row['Accuracy'] = 'N/A'
            row['Completeness'] = 'N/A'
            row['Usability'] = 'N/A'
            updated_rows.append(row)
            continue
        
        print("\n" + "="*80)
        print(f"Tesztazonosító: {row['TestID']}")
        print(f"Kategória: {row['Category']}")
        print(f"Konfiguráció: {row['Configuration']}")
        print(f"Kérés: {row['Query']}")
        print("-"*40)
        print("Válasz (első 500 karakter):")
        print(row['Response'])
        print("-"*40)
        
        # Értékelés bekérése
        accuracy = input("Pontosság (1-5, vagy 's' a kihagyáshoz): ")
        if accuracy.lower() == 's':
            updated_rows.append(row)
            continue
        
        completeness = input("Teljesség (1-5, vagy 's' a kihagyáshoz): ")
        if completeness.lower() == 's':
            updated_rows.append(row)
            continue
        
        usability = input("Használhatóság (1-5, vagy 's' a kihagyáshoz): ")
        if usability.lower() == 's':
            updated_rows.append(row)
            continue
        
        notes = input("Megjegyzések: ")
        
        # Eredmények frissítése
        row['Accuracy'] = accuracy if accuracy.isdigit() else ''
        row['Completeness'] = completeness if completeness.isdigit() else ''
        row['Usability'] = usability if usability.isdigit() else ''
        row['Notes'] = notes
        
        updated_rows.append(row)
    
    # Eredmények mentése
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = reader.fieldnames if 'reader' in locals() else list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    
    print(f"\nAz értékelések mentve a következő fájlba: {results_file}")
    
    # Értékelés összesítése
    evaluated_rows = [row for row in updated_rows if row.get('Accuracy') and row.get('Accuracy') != 'N/A']
    if evaluated_rows:
        print("\nÉrtékelés összesítés:")
        
        # Konfigurációnkénti statisztikák
        config_stats = {}
        for row in evaluated_rows:
            config = row['Configuration']
            if config not in config_stats:
                config_stats[config] = {
                    'total': 0,
                    'accuracy_sum': 0,
                    'completeness_sum': 0,
                    'usability_sum': 0
                }
            
            config_stats[config]['total'] += 1
            if row.get('Accuracy') and row['Accuracy'].isdigit():
                config_stats[config]['accuracy_sum'] += int(row['Accuracy'])
            if row.get('Completeness') and row['Completeness'].isdigit():
                config_stats[config]['completeness_sum'] += int(row['Completeness'])
            if row.get('Usability') and row['Usability'].isdigit():
                config_stats[config]['usability_sum'] += int(row['Usability'])
        
        # Eredmények kiírása
        for config, stats in config_stats.items():
            total = stats['total']
            if total > 0:
                avg_accuracy = stats['accuracy_sum'] / total
                avg_completeness = stats['completeness_sum'] / total
                avg_usability = stats['usability_sum'] / total
                
                print(f"\n{config}:")
                print(f"  Átlagos pontosság: {avg_accuracy:.2f}")
                print(f"  Átlagos teljesség: {avg_completeness:.2f}")
                print(f"  Átlagos használhatóság: {avg_usability:.2f}")

# Ha közvetlenül futtatjuk a fájlt
if __name__ == "__main__":
    print("Budapest Explorer - Értékelő eszköz (javított verzió)")
    print("=" * 60)
    print("API kulcsok beállítva: OPENAI_API_KEY és MAPS_API_KEY")
    
    choice = input("Válassz műveletet:\n1. Tesztek futtatása\n2. Eredmények manuális értékelése\n3. Folytatás a legutóbbi fájlból\nVálasztás: ")
    
    if choice == "1":
        print("\nTesztek futtatása...")
        run_tests()
    elif choice == "2":
        print("\nEredmények manuális értékelése...")
        manual_evaluation()
    elif choice == "3":
        print("\nTesztek folytatása a legutóbbi fájlból...")
        # Legutóbbi fájl keresése
        files = [f for f in os.listdir('.') if f.startswith('test_results_') and f.endswith('.csv')]
        if not files:
            print("Nem található eredményfájl! Új tesztek indítása...")
            run_tests()
        else:
            most_recent_file = max(files)  # A legutolsó fájl (időbélyeg alapján)
            print(f"Folytatás a következő fájlból: {most_recent_file}")
            run_tests(most_recent_file)
    else:
        print("Érvénytelen választás!")
