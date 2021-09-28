import json

test = {}

# Corresponds to each test
test['test_verb_check'] = []

# Corresponds to each part of the test

# Part 0 Test ul/ol tags
test['test_verb_check'].append({
    'input': ['''<ul>
    <li>Performing daily &amp; monthly reconciliations, analysis, break assignment and resolution of reconciliation breaks, daily portfolio valuation processes, account level pricing</li>
    </ul>''', '''<ul>
    <li>Diploma in Accounting or Finance related area (or equivalent certification)</li>
    </ul>'''],
    'output': '''<ul>
    <li>Performing daily &amp; monthly reconciliations, analysis, break assignment and resolution of reconciliation breaks, daily portfolio valuation processes, account level pricing</li>
    </ul>'''
})

# Part 1 Test p tags
test['test_verb_check'].append({
    'input': ['''<p>1. Perform warehouse duty of inbound and outbound activities<br>
              2. Ensure proper cargo checking upon receipt and issuing<br>
              3. Administration of Warehouse Management System (WMS), stock take &amp; cycle count activities<br>
              4. Maintain accurate inventory records<br>
              5. Housekeeping of warehouse area</p>'''],
    'output': '''<p>1. Perform warehouse duty of inbound and outbound activities<br/>
              2. Ensure proper cargo checking upon receipt and issuing<br/>
              3. Administration of Warehouse Management System (WMS), stock take &amp; cycle count activities<br/>
              4. Maintain accurate inventory records<br/>
              5. Housekeeping of warehouse area</p>'''
})

# Part 2 If no verbs are present
test['test_verb_check'].append({
    'input': ['''<ul>
    <li>Diploma in Accounting or Finance related area (or equivalent certification)</li>
    </ul>''']
})

with open('test.txt', 'w') as outfile:
    json.dump(test, outfile)
