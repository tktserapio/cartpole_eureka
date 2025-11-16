from eureka_loop import eureka_search

if __name__ == "__main__":
    best_code, best_score = eureka_search()
    print("\n=== BEST REWARD FUNCTION FOUND ===\n")
    print(best_code)
    print("\nBest score:", best_score)