from django.shortcuts import render
from django.db import connection
import pymysql


def table_view(request):
    # Execute query to select data from table
    with connection.cursor() as cursor:
        cursor.execute("SELECT `id`, `title`, `content`, `sentiment` FROM `temp_items` WHERE 1 LIMIT 1000")
        results = cursor.fetchall()

    # Pass results to template
    context = {'results': results}
    return render(request, 'website/table.html', context)