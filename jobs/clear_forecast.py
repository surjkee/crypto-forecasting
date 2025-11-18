from data.db import get_connection, init_db

def main():
    init_db()
    con = get_connection()
    con.execute("DELETE FROM forecast_hourly;")
    con.close()
    print("✅ Таблиця forecast_hourly очищена.")

if __name__ == "__main__":
    main()
