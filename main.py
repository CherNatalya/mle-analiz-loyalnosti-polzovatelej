from src.processing import DataProcessor


def print_header(title):
    """Вспомогательная функция для заголовков."""
    print("\n" + "-" * 60)
    print(f" {title.upper()} ")
    print("-" * 60)


def main():
    # 1. Инициализация и загрузка данных
    processor = DataProcessor()
    
    print_header("Загрузка и предобработка данных")
    processor.load_purchases()
    processor.clean_data()
    
    # 2. Создание профилей пользователей
    profile = processor.build_profile()
    print(f"\n[INFO] Очищенных заказов: {len(processor.clean):,}")
    print(f"[INFO] Пользователей в профиле: {len(profile):,}")
    print(f"[INFO] Средняя по пользователям выручка с заказа: {profile['avg_revenue_rub'].mean():.2f} ₽")
    print(f"[INFO] Средний чек по всем заказам: {processor.clean['revenue_rub'].mean():.2f} ₽")
    print(f"[INFO] Доля пользователей с 2+ заказами: {profile['is_two'].mean() * 100:.2f}%")

    # 3. Расчет Retention по сегментам
    print_header("Retention по типу первого мероприятия")
    print(processor.retention_by("first_event_type")[["users", "share_two_%"]])

    print_header("Retention по устройству")
    print(processor.retention_by("first_device")[["users", "share_two_%"]])

    print_header("Retention по региону (Топ-10)")
    print(processor.retention_by("first_region", top_n=10)[["users", "share_two_%"]])

    # 4. Проверка гипотез
    print_header("Проверка продуктовых гипотез")
    
    event_hyp = processor.test_event_hypothesis()
    print(f"Гипотеза 1 (спорт > концерты): {'Подтверждена' if event_hyp['confirmed'] else 'Опровергнута'}")
    print(f"  Спорт: {event_hyp['sport_share_%']}%, Концерты: {event_hyp['concerts_share_%']}%")

    region_hyp = processor.test_region_hypothesis()
    print(f"\nГипотеза 2 (крупные регионы > средние): {'Подтверждена' if region_hyp['confirmed'] else 'Не подтверждена'}")
    print(f"  Средний retention в крупных: {region_hyp['large_mean_%']}%")
    print(f"  Максимальный в средних: {region_hyp['medium_max_%']}%")

    # 5. Корреляционный анализ
    print_header("Корреляционный анализ (phi_k)")
    print("Связь признаков с сегментом числа заказов (1, 2, 3-4, 5+):")
    corr_matrix = processor.segment_orders()  # Сохранили в переменную
    print(corr_matrix)

    # 6. Итоговый вывод и рекомендации
    print_header("ИТОГОВЫЙ ОТЧЕТ И РЕКОМЕНДАЦИИ")
    
    # Динамически подтягиваем цифры
    avg_rev_user = profile['avg_revenue_rub'].mean()
    avg_check_all = processor.clean['revenue_rub'].mean()
    share_two = profile['is_two'].mean() * 100
    
    med_rev_1 = profile[~profile['is_two']]['avg_revenue_rub'].median()
    med_rev_2 = profile[profile['is_two']]['avg_revenue_rub'].median()
    med_diff = med_rev_2 - med_rev_1
    
    print(f"• Средняя выручка на пользователя: {avg_rev_user:.2f} ₽ (средний чек по заказам: {avg_check_all:.2f} ₽)")
    print(f"• Доля пользователей с 2+ заказами: {share_two:.2f}%")
    print(f"• Медианный чек: у 1 заказа — {med_rev_1:.2f} ₽, у 2+ заказов — {med_rev_2:.2f} ₽ (разница {med_diff:.2f} ₽)")
    
    print(f"\n• Гипотеза 1 (спорт > концерты): {'Подтверждена' if event_hyp['confirmed'] else 'Опровергнута'}")
    print(f"• Гипотеза 2 (крупные регионы > средние): {'Подтверждена' if region_hyp['confirmed'] else 'Не подтверждена'}")
    
    print("\n• Сильнейшие признаки (phi_k):")
    # Берем значения из уже посчитанной переменной corr_matrix
    print(f"  - avg_days_between: {corr_matrix.loc['avg_days_between', 'orders_segment']}")
    print(f"  - avg_revenue_rub: {corr_matrix.loc['avg_revenue_rub', 'orders_segment']}")
    
    print("\n--- ОГРАНИЧЕНИЯ ---")
    print("1. avg_days_between и avg_revenue_rub известны только после нескольких покупок.")
    print("   Для раннего удержания (в момент 1-го заказа) их использовать нельзя.")
    print("2. Данные охватывают лето–осень 2024 года; выводы по сезонности и долгосрочному удержанию носят оценочный характер.")
    print("3. Тип, устройство, регион, день недели 1-го заказа дают слабую связь.")
    print("   Не стоит использовать как основу для прогнозирования возврата, только как вспомогательные признаки.")
    
    print("\n--- РЕКОМЕНДАЦИИ ---")
    print("1. Повышать медианный чек (сопутствующие товары), а не гнаться за разовыми.")
    print("2. Удерживать 'точки входа' с высокой долей возврата: театр, выставки, концерты, desktop,")
    print("   регионы Шанырский, Светополянский, Широковская, Североярская.")
    print("3. Для крупных сегментов mobile + концерты/другое тестировать промо на 2-й заказ и смежные жанры.")
    print("4. Для сегментов с низкой долей возврата (спорт, ёлки) тестировать рекомендации и промо на смежные жанры.")
    print("\n" + "-" * 60)

if __name__ == "__main__":
    main()