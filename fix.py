# Run this from: C:\Users\likhi\Downloads\TelecomChurn_Final\project\
# It patches streamlit_app.py to fix the ArrowStringArray vs Plotly incompatibility

f = open('app/streamlit_app.py', encoding='utf-8')
c = f.read()
f.close()

old = '    df = get_clean_data().copy()\n    df["Churn_label"] = df["Churn"].map({1: "Churn", 0: "No Churn"})'
new = '    df = get_clean_data().copy()\n    df["Churn_label"] = df["Churn"].map({1: "Churn", 0: "No Churn"}).astype(str)'

if old in c:
    c = c.replace(old, new)
    f = open('app/streamlit_app.py', 'w', encoding='utf-8')
    f.write(c)
    f.close()
    print("Fixed: Churn_label will now use plain str dtype")
else:
    print("ERROR: Could not find the target line. Current Churn_label line:")
    for i, line in enumerate(c.split('\n'), 1):
        if 'Churn_label' in line and 'map' in line:
            print(f"  Line {i}: {line}")
