"""Модуль обработки и анализа данных Яндекс Афиши."""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from phik import phik_matrix
from sqlalchemy import create_engine
import urllib.request

# Загружаем переменные окружения из .env
load_dotenv()


class DataProcessor:
    """Класс для загрузки, обработки, построения профиля и анализа данных."""

    def __init__(self):
        self.purchases_path = "data/afisha_purchases.csv"
        self.tenge_path = "data/final_tickets_tenge_df.csv"
        self.tenge_url = "https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv"
        self.raw = None
        self.clean = None
        self.profile = None


    # Загрузка
    def load_purchases(self):
        """Загружает данные: сначала пробует БД, если не выходит — читает CSV."""
        try:
            # Пробуем получить данные из БД
            self.raw = self._fetch_from_db()
            
            # Сохраняем в CSV
            self.raw.to_csv(self.purchases_path, index=False)
            print(f"Данные успешно выгружены из БД и сохранены в {self.purchases_path}")
            
        except Exception as e:
            # При неудавшемся подключении к БД читаем CSV, если есть
            print(f"Не удалось подключиться к БД ({e}).")
            if os.path.exists(self.purchases_path):
                self.raw = pd.read_csv(self.purchases_path, parse_dates=["order_dt", "order_ts"])
            else:
                raise RuntimeError("Нет доступа к БД, и локальный CSV-файл не найден.")
        
        return self.raw

    def load_tenge_rates(self):
        rates = pd.read_csv(self.tenge_path, parse_dates=["data"])
        rates["rate_per_1_kzt"] = rates["curs"] / rates["nominal"]
        return rates[["data", "rate_per_1_kzt"]]

    def _fetch_from_db(self):
        """Приватный метод для выгрузки данных из БД."""
        user = os.getenv("DB_USER")
        pwd = os.getenv("DB_PWD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        db = os.getenv("DB_NAME")

        engine = create_engine(f"postgresql://{user}:{pwd}@{host}:{port}/{db}")

        query = """
        SELECT
            user_id,
            device_type_canonical,
            order_id,
            created_dt_msk AS order_dt,
            created_ts_msk AS order_ts,
            currency_code,
            revenue,
            tickets_count,
            event_id,
            event_type_main,
            service_name,
            region_name,
            city_name
        FROM afisha.purchases
        JOIN afisha.events  USING (event_id)
        JOIN afisha.city    USING (city_id)
        JOIN afisha.regions USING (region_id)
        WHERE device_type_canonical IN ('mobile', 'desktop')
          AND event_type_main != 'фильм';
        """
        return pd.read_sql_query(query, con=engine)

    # Предобработка
    def convert_currency(self):
        """Приводит выручку к рублю."""
        tenge_rates = self.load_tenge_rates()
        df = self.raw.merge(
            tenge_rates.rename(columns={"data": "order_dt"}),
            on="order_dt",
            how="left"
        )
        df["revenue_rub"] = df["revenue"]
        mask_kzt = df["currency_code"] == "kzt"
        df.loc[mask_kzt, "revenue_rub"] = df.loc[mask_kzt, "revenue"] * df.loc[mask_kzt, "rate_per_1_kzt"]
        return df.drop(columns=["rate_per_1_kzt"])

    def clean_data(self):
        """Типы, фильтрация выбросов и дубликатов, пересчёт интервалов."""
        df = self.convert_currency()

        # Оптимизируем типы
        df["tickets_count"] = pd.to_numeric(df["tickets_count"], downcast="integer")

        # Отсекаем выбросы по 99-му перцентилю
        p99 = df["revenue_rub"].quantile(0.99)
        df = df[(df["revenue_rub"] > 0) & (df["revenue_rub"] <= p99)].copy()

        # Удаляем дубликаты
        key_cols = ["user_id", "order_ts", "device_type_canonical",
                    "service_name", "event_id", "revenue_rub"]
        df = (df.sort_values("order_ts")
                .drop_duplicates(subset=key_cols, keep="first")
                .reset_index(drop=True))

        # Пересчитываем интервалы
        df = df.sort_values(["user_id", "order_ts"])
        df["days_since_prev"] = (df.groupby("user_id")["order_ts"]
                                    .diff().dt.days.astype("float32"))
        df = df.sort_values("order_ts").reset_index(drop=True)

        self.clean = df
        return df

    # Профиль
    def build_profile(self):
        """Строит профиль пользователя и фильтрует аномалии."""
        profile = (
            self.clean.groupby("user_id")
            .agg(
                first_order_dt=("order_ts", "min"),
                last_order_dt=("order_ts", "max"),
                first_device=("device_type_canonical", "first"),
                first_region=("region_name", "first"),
                first_event_type=("event_type_main", "first"),
                total_orders=("order_id", "count"),
                avg_revenue_rub=("revenue_rub", "mean"),
                avg_days_between=("days_since_prev", "mean"),
            )
            .reset_index()
        )
        # Бинарный признак: 2 и более заказа
        profile["is_two"] = profile["total_orders"] >= 2

        # Фильтрация
        profile = profile[profile["total_orders"] <= 1000]

        # День недели первого заказа
        profile["first_dow"] = profile["first_order_dt"].dt.day_name()
        
        self.profile = profile.copy()
        return self.profile

    # Аналитика
    def retention_by(self, column, top_n=None):
        """Рассчитывает Retention по сегментам."""
        grp = (self.profile.groupby(column)["is_two"]
               .agg(["count", "mean"])
               .rename(columns={"count": "users", "mean": "share_two"}))
        grp["share_two_%"] = (grp["share_two"] * 100).round(2)
        
        if top_n:
            grp = grp.sort_values("users", ascending=False).head(top_n)
            
        return grp.sort_values("share_two", ascending=False)

    def test_event_hypothesis(self):
        """Проверка гипотезы: люди с первым заказом на спорт возвращаются чаще концертов."""
        ret = self.retention_by("first_event_type")
        sport = float(ret.loc["спорт", "share_two_%"])
        concerts = float(ret.loc["концерты", "share_two_%"])
        return {
            "sport_share_%": sport,
            "concerts_share_%": concerts,
            "confirmed": bool(sport > concerts),
        }

    def test_region_hypothesis(self):
        """Проверка гипотезы: крупные регионы возвращаются чаще."""
        ret = self.retention_by("first_region")
        large = ret[ret["users"] >= 1000]["share_two_%"]
        medium = ret[(ret["users"] >= 400) & (ret["users"] < 1000)]["share_two_%"]
        return {
            "large_mean_%": round(float(large.mean()), 2),
            "medium_max_%": round(float(medium.max()), 2),
            "confirmed": bool(len(large) and len(medium) and large.mean() > medium.max()),
        }

    def segment_orders(self):
        """Корреляционный анализ (phi_k) по сегментам."""
        def _segment_orders(n):
            if n == 1: return "1"
            if n == 2: return "2"
            if n <= 4: return "3-4"
            return "5+"

        profile = self.profile.copy()
        profile["orders_segment"] = profile["total_orders"].apply(_segment_orders)
        
        corr_cols = ["orders_segment", "first_device", "first_region",
                "first_event_type", "avg_revenue_rub",
                "avg_days_between", "first_dow"]
                
        corr_matrix = profile[corr_cols].phik_matrix(
            interval_cols=["avg_revenue_rub", "avg_days_between"]
        )
        return (corr_matrix[["orders_segment"]]
                .drop(index="orders_segment")
                .sort_values("orders_segment", ascending=False)
                .round(3))
