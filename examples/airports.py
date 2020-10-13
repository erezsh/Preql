"""Start a Preql interactive environment with an airports table, based on a JSON url.

Uses pandas for reading and handling the JSON file.
"""

import pandas as pd
from preql import Preql

AIRPORTS_JSON_URL = 'https://gist.githubusercontent.com/tdreyno/4278655/raw/7b0762c09b519f40397e4c3e100b097d861f5588/airports.json'

airports = pd.read_json(AIRPORTS_JSON_URL)

p = Preql()
p.import_pandas(airports_full=airports)
p('''
    airports = airports_full{... !url !tz !phone !email !type}

    // All airports sorted by elevation (highest first)
    airports_by_highest = airports order {^elev}

    // Count of airports in each country, sorted by highest
    airports_by_country = airports {country => airport_count: count()} order {^airport_count}
''')
p.start_repl()