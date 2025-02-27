package com.library.entity;

import java.time.LocalDateTime;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name = "borrowings")
public class Borrowing {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne
    @JoinColumn(name = "book_id", nullable = false)
    private Book book;

    private LocalDateTime borrowDate;

    private LocalDateTime dueDate;

    private LocalDateTime returnDate;

    private String status; // BORROWED, RETURNED, OVERDUE

    private String notes;

    // Default Constructor
    public Borrowing() {
    }

    // All-args Constructor
    public Borrowing(Long id, User user, Book book, LocalDateTime borrowDate,
                    LocalDateTime dueDate, LocalDateTime returnDate, String status, String notes) {
        this.id = id;
        this.user = user;
        this.book = book;
        this.borrowDate = borrowDate;
        this.dueDate = dueDate;
        this.returnDate = returnDate;
        this.status = status;
        this.notes = notes;
    }

    // Getters
    public Long getId() {
        return id;
    }

    public User getUser() {
        return user;
    }

    public Book getBook() {
        return book;
    }

    public LocalDateTime getBorrowDate() {
        return borrowDate;
    }

    public LocalDateTime getDueDate() {
        return dueDate;
    }

    public LocalDateTime getReturnDate() {
        return returnDate;
    }

    public String getStatus() {
        return status;
    }

    public String getNotes() {
        return notes;
    }

    // Setters
    public void setId(Long id) {
        this.id = id;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public void setBook(Book book) {
        this.book = book;
    }

    public void setBorrowDate(LocalDateTime borrowDate) {
        this.borrowDate = borrowDate;
    }

    public void setDueDate(LocalDateTime dueDate) {
        this.dueDate = dueDate;
    }

    public void setReturnDate(LocalDateTime returnDate) {
        this.returnDate = returnDate;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public void setNotes(String notes) {
        this.notes = notes;
    }

    // equals and hashCode methods
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Borrowing borrowing = (Borrowing) o;
        return id != null && id.equals(borrowing.id);
    }

    @Override
    public int hashCode() {
        return id != null ? id.hashCode() : 0;
    }

    // toString method
    @Override
    public String toString() {
        return "Borrowing{" +
                "id=" + id +
                ", user=" + user.getId() +
                ", book=" + book.getId() +
                ", borrowDate=" + borrowDate +
                ", dueDate=" + dueDate +
                ", returnDate=" + returnDate +
                ", status='" + status + '\'' +
                ", notes='" + notes + '\'' +
                '}';
    }
} 