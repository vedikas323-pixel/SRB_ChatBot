import camelot

print("Extracting tables from PDF...")
tables = camelot.read_pdf("SRB.pdf", pages="all")
tables.export("tables.csv", f="csv")
print("Done! Check tables.csv in the project folder.")