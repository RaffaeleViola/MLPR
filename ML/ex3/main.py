


import sys




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dic"]
    months = dict()
    cities = dict()
    n_births = 0

    with open(sys.argv[1], "r") as f:
        for line in f:
            words = line.split()
            month = int(words[3].split("/")[1]) - 1
            months.setdefault(months_list[month], 0)
            months[months_list[month]] += 1
            cities.setdefault(words[2], 0)
            cities[words[2]] += 1
            n_births += 1

    avg_births_per_city = n_births / len(cities.keys())
    print("Births per city")
    for key, value in cities.items():
        print(f"\t%s: %d", key, value)

    print("Births per month")
    for key, value in months.items():
        if value == 0:
            continue
        print(f"\t%s: %d", key, value)

    print(f"Average: %f", avg_births_per_city)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
